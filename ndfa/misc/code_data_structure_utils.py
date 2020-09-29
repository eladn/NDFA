import json
import argparse
import dataclasses
import numpy as np
from functools import reduce
from typing_extensions import Protocol
from typing import List, Dict, Tuple, Set, Optional, Any, Union

from ndfa.misc.code_data_structure_api import SerASTNodeType, SerASTNode, SerMethodAST, SerMethod, SerPDGNode, \
    SerMethodPDG, SerPDGControlFlowEdge, SerPDGDataDependencyEdge, SerControlScopeType
from ndfa.misc.iter_raw_extracted_data_files import RawExtractedExample


__all__ = [
    'get_pdg_node_tokenized_expression', 'find_all_simple_names_in_sub_ast', 'get_symbol_idxs_used_in_logging_call',
    'traverse_pdg', 'get_all_pdg_simple_paths']


def get_pdg_node_tokenized_expression(method: SerMethod, pdg_node: SerPDGNode):
    return method.code.tokenized[
        pdg_node.code_sub_token_range_ref.begin_token_idx:
        pdg_node.code_sub_token_range_ref.end_token_idx+1]


def find_all_simple_names_in_sub_ast(ast_node: SerASTNode, method_ast: SerMethodAST) -> Set[str]:
    if ast_node.type == SerASTNodeType.SIMPLE_NAME:
        assert ast_node.identifier is not None
        return {ast_node.identifier}
    children = [method_ast.nodes[child_idx] for child_idx in ast_node.children_idxs]
    return reduce(
        lambda union, child: union.union(
            find_all_simple_names_in_sub_ast(child, method_ast=method_ast)), children, set())


def get_symbol_idxs_used_in_logging_call(example: RawExtractedExample) -> Tuple[int, ...]:
    if example.logging_call.pdg_node_idx is None:
        return ()
    logging_call_pdg_node = example.method_pdg.pdg_nodes[example.logging_call.pdg_node_idx]

    # TODO: this is a very primitive way to remove the "LOG" prefix.
    #  We should implement it correctly in the JavaExtractor. The correct impl (there) should be to remove
    #  the ExpressionBase of the method call that does not apprear in the method call params.
    logging_call_ast_node = example.method_ast.nodes[example.logging_call.ast_node_idx]
    logging_call_ast_children_nodes = [
        example.method_ast.nodes[child_idx] for child_idx in logging_call_ast_node.children_idxs]
    assert sum(int(child.type == SerASTNodeType.SIMPLE_NAME)
               for child in logging_call_ast_children_nodes) == 1
    assert (logging_call_ast_children_nodes[0].type == SerASTNodeType.SIMPLE_NAME) ^ \
           (len(logging_call_ast_children_nodes) >= 2 and
            logging_call_ast_children_nodes[1].type == SerASTNodeType.SIMPLE_NAME)
    symbol_names_to_remove = find_all_simple_names_in_sub_ast(
        logging_call_ast_children_nodes[0], method_ast=example.method_ast)

    return tuple(
        set(symbol_ref.symbol_idx for symbol_ref in logging_call_pdg_node.symbols_use_def_mut.use.must
            if symbol_ref.symbol_name.lower().strip('_') not in {'log', 'logger'} and
            symbol_ref.symbol_name not in symbol_names_to_remove) |
        set(symbol_ref.symbol_idx for symbol_ref in logging_call_pdg_node.symbols_use_def_mut.use.may
            if symbol_ref.symbol_name.lower().strip('_') not in {'log', 'logger'} and
            symbol_ref.symbol_name not in symbol_names_to_remove))


@dataclasses.dataclass
class PDGTraversePreVisitFnRes:
    trim_branch_traversal: bool = False


@dataclasses.dataclass
class PDGTraversePostVisitFnRes:
    val: Any
    stop_traversal: bool = False


class PDGTraversePreVisitFn(Protocol):
    def __call__(self, pdg_node_idx: int, already_visited_before: bool) -> Optional[PDGTraversePreVisitFnRes]: ...


class PDGTraversePostVisitFn(Protocol):
    def __call__(self, pdg_node_idx: int, already_visited_before: bool,
                 successor_results: Optional[Tuple[Union[SerPDGControlFlowEdge, SerPDGDataDependencyEdge], Any]]) \
            -> Optional[PDGTraversePostVisitFnRes]: ...


def traverse_pdg(method_pdg: SerMethodPDG, src_node_idx: int, tgt_node_idx: int,
                 pre_visitor_fn: Optional[PDGTraversePreVisitFn] = None,
                 post_visitor_fn: Optional[PDGTraversePostVisitFn] = None,
                 traverse_control_flow_edges: bool = True, traverse_data_dependency_edges: bool = True,
                 revisit_nodes_on_different_branches: bool = True, propagate_results: bool = True,
                 reverse: bool = False, twice_loop_traversal: bool = False) -> Any:
    if twice_loop_traversal:
        # TODO: implement the option to traverse loop twice.
        raise NotImplementedError('traverse_pdg(): `twice_loop_traversal` option not implemented yet!')
    occurrences_in_dfs_stack_count = np.zeros(len(method_pdg.pdg_nodes), dtype=np.int)
    already_visited = np.zeros(len(method_pdg.pdg_nodes), dtype=np.bool)
    traversal_control = argparse.Namespace(stop_traversal=False)
    def _dfs(cur_node_idx: int) -> Any:
        cur_node = method_pdg.pdg_nodes[cur_node_idx]
        nr_allowed_visits_in_path_for_cur_node = 1
        loop_update_condition_control_scope_types = {
            SerControlScopeType.LOOP_CONDITION.value,
            SerControlScopeType.LOOP_UPDATE.value,
            SerControlScopeType.LOOP_UPDATE_AND_CONDITION.value}
        if method_pdg.control_scopes[cur_node.belongs_to_control_scopes_idxs[-1]].type.value in \
                loop_update_condition_control_scope_types:
            nr_allowed_visits_in_path_for_cur_node = 2
        assert occurrences_in_dfs_stack_count[cur_node_idx] <= nr_allowed_visits_in_path_for_cur_node
        if occurrences_in_dfs_stack_count[cur_node_idx] >= nr_allowed_visits_in_path_for_cur_node:
            return None
        if not revisit_nodes_on_different_branches and already_visited[cur_node_idx]:
            return None
        occurrences_in_dfs_stack_count[cur_node_idx] += 1
        successor_results = None
        if cur_node_idx != tgt_node_idx and not traversal_control.stop_traversal:
            pre_visitor_res = None
            if pre_visitor_fn is not None:
                pre_visitor_res = pre_visitor_fn(cur_node_idx, already_visited[cur_node_idx])
            if pre_visitor_res is None or not pre_visitor_res.trim_branch_traversal:
                if propagate_results and post_visitor_fn is not None:
                    successor_results = []
                edges = []
                if traverse_control_flow_edges:
                    edges += cur_node.control_flow_in_edges if reverse else cur_node.control_flow_out_edges
                if traverse_data_dependency_edges:
                    edges += cur_node.data_dependency_in_edges if reverse else cur_node.data_dependency_out_edges
                for edge in edges:
                    cur_successor_res = _dfs(edge.pgd_node_idx)
                    if propagate_results and post_visitor_fn is not None:
                        successor_results.append((edge, cur_successor_res))
                    if traversal_control.stop_traversal:
                        break
        post_visitor_res = None
        if post_visitor_fn is not None:
            post_visitor_res = post_visitor_fn(cur_node_idx, already_visited[cur_node_idx], successor_results)
        if post_visitor_res is not None and post_visitor_res.stop_traversal:
            traversal_control.stop_traversal = True
        occurrences_in_dfs_stack_count[cur_node_idx] -= 1
        if occurrences_in_dfs_stack_count[cur_node_idx] == 0:
            already_visited[cur_node_idx] = True
        return None if post_visitor_res is None else post_visitor_res.val
    return _dfs(src_node_idx)


@dataclasses.dataclass
class _PathNode:
    node_idx: int
    edge: Optional[Union[SerPDGControlFlowEdge, SerPDGDataDependencyEdge]]

    def __lt__(self, other):
        assert isinstance(other, _PathNode)
        if self.node_idx != other.node_idx:
            return self.node_idx < other.node_idx
        return hash(self) < hash(other)

    def __hash__(self):
        return hash((self.node_idx, None if self.edge is None else json.dumps(self.edge.to_dict(), sort_keys=True)))


def get_all_pdg_simple_paths(
        method_pdg: SerMethodPDG, src_pdg_node_idx: int, tgt_pdg_node_idx: int,
        control_flow: bool = True, data_dependency: bool = False, max_nr_paths: Optional[int] = None) \
        -> Optional[List[Tuple[_PathNode, ...]]]:
    traverse_ctrl = argparse.Namespace(max_nr_paths_found=0)
    simple_paths_from_node_to_tgt: Dict[int, Set[Tuple[_PathNode, ...]]] = {}


    def pre_visitor(pdg_node_idx: int, already_visited_before: bool) -> PDGTraversePreVisitFnRes:
        return PDGTraversePreVisitFnRes(trim_branch_traversal=pdg_node_idx in simple_paths_from_node_to_tgt)

    def post_visitor(
            pdg_node_idx: int, already_visited_before: bool,
            successor_results: Optional[Tuple[Union[SerPDGControlFlowEdge, SerPDGDataDependencyEdge], Any]]) \
            -> Optional[PDGTraversePostVisitFnRes]:
        if pdg_node_idx == tgt_pdg_node_idx:
            simple_paths_from_node_to_tgt[pdg_node_idx] = {(_PathNode(pdg_node_idx, None),)}
        else:
            assert pdg_node_idx in simple_paths_from_node_to_tgt or successor_results is not None
            if successor_results is not None:
                new_paths = {
                    (_PathNode(pdg_node_idx, successor_result[0]),) + path_to_successor
                    for successor_result in successor_results
                    if successor_result is not None and successor_result[1] is not None
                    for path_to_successor in successor_result[1][1]}
                if pdg_node_idx not in simple_paths_from_node_to_tgt:
                    simple_paths_from_node_to_tgt[pdg_node_idx] = set()
                simple_paths_from_node_to_tgt[pdg_node_idx].update(new_paths)
                traverse_ctrl.max_nr_paths_found = max(
                    traverse_ctrl.max_nr_paths_found, len(simple_paths_from_node_to_tgt[pdg_node_idx]))
        return PDGTraversePostVisitFnRes(
            val=(len(simple_paths_from_node_to_tgt[pdg_node_idx]), simple_paths_from_node_to_tgt[pdg_node_idx]),
            stop_traversal=max_nr_paths is not None and traverse_ctrl.max_nr_paths_found > max_nr_paths)

    nr_paths, paths = traverse_pdg(
        method_pdg=method_pdg, src_node_idx=src_pdg_node_idx, tgt_node_idx=tgt_pdg_node_idx,
        post_visitor_fn=post_visitor, pre_visitor_fn=pre_visitor, traverse_control_flow_edges=control_flow,
        traverse_data_dependency_edges=data_dependency, revisit_nodes_on_different_branches=True)

    paths = sorted(list(paths))
    if max_nr_paths is not None and nr_paths >= max_nr_paths:
        return None
    return paths

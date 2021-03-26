import json
import argparse
import itertools
import dataclasses
import numpy as np
from enum import Enum
from functools import reduce
from collections import defaultdict
from typing_extensions import Protocol
from typing import List, Dict, Tuple, Set, Optional, Any, Union, Iterable, Generic, TypeVar

from ndfa.misc.code_data_structure_api import SerASTNodeType, SerASTNode, SerMethodAST, SerMethod, SerPDGNode, \
    SerMethodPDG, SerPDGControlFlowEdge, SerPDGDataDependencyEdge, SerControlScopeType, SerToken
from ndfa.misc.iter_raw_extracted_data_files import RawExtractedExample


__all__ = [
    'get_pdg_node_tokenized_expression', 'get_pdg_node_expression_str', 'get_ast_node_tokenized_expression',
    'get_ast_node_expression_str', 'find_all_simple_names_in_sub_ast', 'get_symbol_idxs_used_in_logging_call',
    'traverse_pdg', 'get_all_pdg_simple_paths', 'get_all_ast_paths', 'ASTPaths', 'ASTLeaf2InnerNodePathNode',
    'ASTLeaf2LeafPathNode', 'ASTNodeIdxType', 'traverse_ast', 'ast_node_to_str', 'print_ast']


def get_pdg_node_tokenized_expression(method: SerMethod, pdg_node: SerPDGNode) -> List[SerToken]:
    return method.code.tokenized[
        pdg_node.code_sub_token_range_ref.begin_token_idx:
        pdg_node.code_sub_token_range_ref.end_token_idx+1]


def get_pdg_node_expression_str(method: SerMethod, pdg_node: SerPDGNode) -> str:
    return method.code.code_str[
        method.code.tokenized[
            pdg_node.code_sub_token_range_ref.begin_token_idx].position_range_in_code_snippet_str.begin_idx:
        method.code.tokenized[
            pdg_node.code_sub_token_range_ref.end_token_idx].position_range_in_code_snippet_str.end_idx + 1]


def get_ast_node_tokenized_expression(method: SerMethod, ast_node: SerASTNode) -> List[SerToken]:
    return method.code.tokenized[
        ast_node.code_sub_token_range_ref.begin_token_idx:
        ast_node.code_sub_token_range_ref.end_token_idx+1]


def get_ast_node_expression_str(method: SerMethod, ast_node: SerASTNode) -> str:
    return method.code.code_str[
        method.code.tokenized[
            ast_node.code_sub_token_range_ref.begin_token_idx].position_range_in_code_snippet_str.begin_idx:
        method.code.tokenized[
            ast_node.code_sub_token_range_ref.end_token_idx].position_range_in_code_snippet_str.end_idx + 1]


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


T = TypeVar('T')


@dataclasses.dataclass
class PDGTraversePostVisitFnRes(Generic[T]):
    val: T
    stop_traversal: bool = False


class PDGTraversePreVisitFn(Protocol):
    def __call__(self, pdg_node_idx: int, already_visited_before: bool) -> Optional[PDGTraversePreVisitFnRes]: ...


PDGEdge = Union[SerPDGControlFlowEdge, SerPDGDataDependencyEdge]
PDGEdgesGroup = List[PDGEdge]


class PDGTraversePostVisitFn(Protocol[T]):
    def __call__(self, pdg_node_idx: int, already_visited_before: bool,
                 successor_results: Optional[List[Tuple[PDGEdgesGroup, T]]]) \
            -> Optional[PDGTraversePostVisitFnRes[T]]: ...


def traverse_pdg(method_pdg: SerMethodPDG, src_node_idx: int, tgt_node_idx: int,
                 pre_visitor_fn: Optional[PDGTraversePreVisitFn] = None,
                 post_visitor_fn: Optional[PDGTraversePostVisitFn[T]] = None,
                 traverse_control_flow_edges: bool = True, traverse_data_dependency_edges: bool = True,
                 max_nr_data_dependency_edges_in_path: Optional[int] = None,
                 max_nr_control_flow_edges_in_path: Optional[int] = None,
                 group_different_edges_of_single_nodes_pair_in_same_path: bool = False,
                 revisit_nodes_on_different_branches: bool = True, propagate_results: bool = True,
                 reverse: bool = False, twice_loop_traversal: bool = False,
                 remove_data_dependency_edges_from_pdg_nodes_idxs: Optional[Set[int]] = None) -> Any:
    if twice_loop_traversal:
        # TODO: implement the option to traverse loop twice.
        raise NotImplementedError('traverse_pdg(): `twice_loop_traversal` option not implemented yet!')
    nodes_occurrences_in_dfs_stack_count = np.zeros(len(method_pdg.pdg_nodes), dtype=np.int)
    nr_edges_occurrences_in_dfs_stack_by_type = {'data_dependency': 0, 'control_flow': 0}
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
        assert nodes_occurrences_in_dfs_stack_count[cur_node_idx] <= nr_allowed_visits_in_path_for_cur_node
        if nodes_occurrences_in_dfs_stack_count[cur_node_idx] >= nr_allowed_visits_in_path_for_cur_node:
            return None
        if not revisit_nodes_on_different_branches and already_visited[cur_node_idx]:
            return None
        nodes_occurrences_in_dfs_stack_count[cur_node_idx] += 1
        successor_results = None
        if cur_node_idx != tgt_node_idx and not traversal_control.stop_traversal:
            pre_visitor_res = None
            if pre_visitor_fn is not None:
                pre_visitor_res = pre_visitor_fn(cur_node_idx, already_visited[cur_node_idx])
            if pre_visitor_res is None or not pre_visitor_res.trim_branch_traversal:
                if propagate_results and post_visitor_fn is not None:
                    successor_results = []
                edges_by_tgt_node_idx: Dict[int, PDGEdgesGroup] = \
                    defaultdict(list)
                if traverse_control_flow_edges:
                    control_flow_edges = cur_node.control_flow_in_edges if reverse else cur_node.control_flow_out_edges
                    for edge in control_flow_edges:
                        edges_by_tgt_node_idx[edge.pgd_node_idx].append(edge)
                if traverse_data_dependency_edges and \
                        (remove_data_dependency_edges_from_pdg_nodes_idxs is None or
                         cur_node_idx not in remove_data_dependency_edges_from_pdg_nodes_idxs):
                    data_dependency_edges = cur_node.data_dependency_in_edges if reverse else cur_node.data_dependency_out_edges
                    for edge in data_dependency_edges:
                        if remove_data_dependency_edges_from_pdg_nodes_idxs is None or \
                                edge.pgd_node_idx not in remove_data_dependency_edges_from_pdg_nodes_idxs:
                            edges_by_tgt_node_idx[edge.pgd_node_idx].append(edge)
                if group_different_edges_of_single_nodes_pair_in_same_path:
                    edges_grouped = list(edges_by_tgt_node_idx.values())
                else:
                    edges_grouped = [[edge] for edges_by_tgt in edges_by_tgt_node_idx.values() for edge in edges_by_tgt]
                for edges_group in edges_grouped:
                    edge_types_in_group = set((
                        'data_dependency' if isinstance(edge, SerPDGDataDependencyEdge) else 'control_flow'
                        for edge in edges_group))
                    sole_edge_type = None
                    if edge_types_in_group == {'data_dependency'}:
                        sole_edge_type = 'data_dependency'
                        if max_nr_data_dependency_edges_in_path is not None and \
                                max_nr_data_dependency_edges_in_path <= \
                                nr_edges_occurrences_in_dfs_stack_by_type['data_dependency']:
                            continue
                    elif edge_types_in_group == {'control_flow'}:
                        sole_edge_type = 'control_flow'
                        if max_nr_control_flow_edges_in_path is not None and \
                            max_nr_control_flow_edges_in_path <= \
                            nr_edges_occurrences_in_dfs_stack_by_type['control_flow']:
                            continue
                    if sole_edge_type is not None:
                        nr_edges_occurrences_in_dfs_stack_by_type[sole_edge_type] += 1
                    cur_successor_res = _dfs(edges_group[0].pgd_node_idx)
                    if sole_edge_type is not None:
                        nr_edges_occurrences_in_dfs_stack_by_type[sole_edge_type] -= 1
                    if propagate_results and post_visitor_fn is not None:
                        successor_results.append((edges_group, cur_successor_res))
                    if traversal_control.stop_traversal:
                        break
        post_visitor_res = None
        if post_visitor_fn is not None:
            post_visitor_res = post_visitor_fn(cur_node_idx, already_visited[cur_node_idx], successor_results)
        if post_visitor_res is not None and post_visitor_res.stop_traversal:
            traversal_control.stop_traversal = True
        nodes_occurrences_in_dfs_stack_count[cur_node_idx] -= 1
        if nodes_occurrences_in_dfs_stack_count[cur_node_idx] == 0:
            already_visited[cur_node_idx] = True
        return None if post_visitor_res is None else post_visitor_res.val
    return _dfs(src_node_idx)


@dataclasses.dataclass
class _PDGPathNode:
    node_idx: int
    edges: Tuple[PDGEdge, ...] = ()

    def __lt__(self, other):
        assert isinstance(other, _PDGPathNode)
        if self.node_idx != other.node_idx:
            return self.node_idx < other.node_idx
        return hash(self) < hash(other)

    def __hash__(self):
        return hash((self.node_idx, None if self.edges is None else tuple(json.dumps(edge.to_dict(), sort_keys=True) for edge in self.edges)))


_PDGPath = Tuple[_PDGPathNode, ...]


@dataclasses.dataclass(unsafe_hash=True, order=True)
class PDGPathWithEdgesCountByType:
    path: _PDGPath
    nr_data_dependency_edges: int
    nr_control_flow_edges: int


def get_all_pdg_simple_paths(
        method_pdg: SerMethodPDG, src_pdg_node_idx: int, tgt_pdg_node_idx: int,
        control_flow: bool = True, data_dependency: bool = False,
        max_nr_paths: Optional[int] = None,
        max_nr_data_dependency_edges_in_path: Optional[int] = None,
        max_nr_control_flow_edges_in_path: Optional[int] = None,
        group_different_edges_of_single_nodes_pair_in_same_path: bool = False,
        remove_data_dependency_edges_from_pdg_nodes_idxs: Optional[Set[int]] = None) \
        -> Optional[List[_PDGPath]]:
    traverse_ctrl = argparse.Namespace(max_nr_paths_found=0)
    simple_paths_from_node_to_tgt: Dict[int, Set[PDGPathWithEdgesCountByType]] = {}

    def pre_visitor(pdg_node_idx: int, already_visited_before: bool) -> PDGTraversePreVisitFnRes:
        return PDGTraversePreVisitFnRes(trim_branch_traversal=pdg_node_idx in simple_paths_from_node_to_tgt)

    def post_visitor(
            pdg_node_idx: int, already_visited_before: bool,
            successor_results: Optional[List[Tuple[PDGEdgesGroup,
                                                   Tuple[int, Set[PDGPathWithEdgesCountByType]]]]]) \
            -> Optional[PDGTraversePostVisitFnRes[Tuple[int, Set[PDGPathWithEdgesCountByType]]]]:
        if pdg_node_idx == tgt_pdg_node_idx:
            simple_paths_from_node_to_tgt[pdg_node_idx] = {PDGPathWithEdgesCountByType(
                path=(_PDGPathNode(pdg_node_idx, ()),), nr_control_flow_edges=0, nr_data_dependency_edges=0)}
        else:
            assert pdg_node_idx in simple_paths_from_node_to_tgt or successor_results is not None
            if successor_results is not None:
                new_paths = {
                    PDGPathWithEdgesCountByType(
                        path=(_PDGPathNode(pdg_node_idx, tuple(successor_result[0])),) + path_to_successor.path,
                        nr_data_dependency_edges=path_to_successor.nr_data_dependency_edges + int(all(isinstance(edge, SerPDGDataDependencyEdge) for edge in successor_result[0])),
                        nr_control_flow_edges=path_to_successor.nr_control_flow_edges + int(all(isinstance(edge, SerPDGControlFlowEdge) for edge in successor_result[0])))
                    for successor_result in successor_results
                    if successor_result is not None and successor_result[1] is not None
                    for path_to_successor in successor_result[1][1]}
                new_paths = {
                    path for path in new_paths
                    if (max_nr_data_dependency_edges_in_path is None or
                        path.nr_data_dependency_edges <= max_nr_data_dependency_edges_in_path) and
                       (max_nr_control_flow_edges_in_path is None or
                        path.nr_control_flow_edges <= max_nr_control_flow_edges_in_path)}
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
        post_visitor_fn=post_visitor, pre_visitor_fn=pre_visitor,
        traverse_control_flow_edges=control_flow, traverse_data_dependency_edges=data_dependency,
        # We enforce max #edges in this method; enforcing it in the traversal itself would harm correctness,
        # as a node might be traversed only once (pre_visitor trims deepening for already-visited nodes).
        max_nr_data_dependency_edges_in_path=None,
        max_nr_control_flow_edges_in_path=None,
        group_different_edges_of_single_nodes_pair_in_same_path=group_different_edges_of_single_nodes_pair_in_same_path,
        revisit_nodes_on_different_branches=True,
        remove_data_dependency_edges_from_pdg_nodes_idxs=remove_data_dependency_edges_from_pdg_nodes_idxs)

    paths = sorted(list(paths))
    paths = [path.path for path in paths]
    if max_nr_paths is not None and nr_paths >= max_nr_paths:
        return None
    return paths


ASTNodeIdxType = int


@dataclasses.dataclass(frozen=True)
class ASTLeaf2InnerNodePathNode:
    ast_node_idx: ASTNodeIdxType
    child_place_in_parent: Optional[int]


@dataclasses.dataclass(frozen=True)
class ASTLeaf2LeafPathNode:
    ast_node_idx: ASTNodeIdxType
    child_place_in_parent: Optional[int]
    direction: 'Direction'

    class Direction(Enum):
        UP = 'UP'
        DOWN = 'DOWN'
        COMMON = 'COMMON'


@dataclasses.dataclass(frozen=True)
class ASTPaths:
    leaf_to_leaf_paths: Dict[Tuple[ASTNodeIdxType, ASTNodeIdxType], Tuple[ASTLeaf2LeafPathNode, ...]]
    leaves_pair_common_ancestor: Dict[Tuple[ASTNodeIdxType, ASTNodeIdxType], ASTNodeIdxType]
    leaf_to_root_paths: Dict[ASTNodeIdxType, Tuple[ASTLeaf2InnerNodePathNode, ...]]
    leaves_sequence: Tuple[ASTNodeIdxType, ...]
    postorder_traversal_sequence: Tuple[ASTNodeIdxType, ...]
    siblings_sequences: Dict[ASTNodeIdxType, Tuple[ASTNodeIdxType, ...]]
    nodes_depth: Dict[ASTNodeIdxType, int]
    subtree_indices_range: Tuple[ASTNodeIdxType, ASTNodeIdxType]


def get_all_ast_paths(
        method_ast: SerMethodAST,
        sub_ast_root_node_idx: ASTNodeIdxType = 0,
        subtrees_to_ignore: Optional[Set[int]] = None,
        verify_preorder_indexing: bool = False) -> ASTPaths:
    all_ast_leaf_to_leaf_paths: Dict[Tuple[ASTNodeIdxType, ASTNodeIdxType], Tuple[ASTLeaf2LeafPathNode, ...]] = {}
    leaves_pair_common_ancestor: Dict[Tuple[ASTNodeIdxType, ASTNodeIdxType], ASTNodeIdxType] = {}
    leaves_sequence: List[ASTNodeIdxType] = []
    postorder_traversal_sequence: List[ASTNodeIdxType] = []
    siblings_sequences: Dict[ASTNodeIdxType, Tuple[ASTNodeIdxType, ...]] = {}
    nodes_depth: Dict[ASTNodeIdxType, int] = {}
    if subtrees_to_ignore is None:
        subtrees_to_ignore = set()
    assert sub_ast_root_node_idx not in subtrees_to_ignore
    if verify_preorder_indexing:
        # just a sanity-check to ensure the indexing method (pre-order).
        # pre-order indexing is later assumed for calculating the field `subtree_indices_range`.
        all_subtree_indices: Set[ASTNodeIdxType] = set()

    def aux_recursive_ast_traversal(
            current_node_idx: ASTNodeIdxType, depth: int = 0) \
            -> List[Tuple[ASTLeaf2InnerNodePathNode, ...]]:
        nodes_depth[current_node_idx] = depth
        if verify_preorder_indexing:
            all_subtree_indices.add(current_node_idx)
        current_node_children_idxs = tuple(
            child_node_idx
            for child_node_idx in method_ast.nodes[current_node_idx].children_idxs
            if child_node_idx not in subtrees_to_ignore)
        if len(current_node_children_idxs) == 0:  # leaf
            leaves_sequence.append(current_node_idx)
            postorder_traversal_sequence.append(current_node_idx)
            return [()]

        if len(current_node_children_idxs) > 1:
            assert current_node_idx not in siblings_sequences
            siblings_sequences[current_node_idx] = current_node_children_idxs

        inner_upward_paths_from_leaves_to_children = [
            aux_recursive_ast_traversal(child_node_idx, depth=depth + 1)
            for child_node_idx in current_node_children_idxs]
        postorder_traversal_sequence.append(current_node_idx)

        for left_child_place in range(len(inner_upward_paths_from_leaves_to_children)):
            for right_child_place in range(left_child_place + 1, len(inner_upward_paths_from_leaves_to_children)):
                ret_from_left_child = inner_upward_paths_from_leaves_to_children[left_child_place]
                ret_from_right_child = inner_upward_paths_from_leaves_to_children[right_child_place]
                for left_path in ret_from_left_child:
                    for right_path in ret_from_right_child:
                        left_path += (ASTLeaf2InnerNodePathNode(
                            ast_node_idx=current_node_children_idxs[left_child_place],
                            child_place_in_parent=left_child_place),)
                        right_path += (ASTLeaf2InnerNodePathNode(
                            ast_node_idx=current_node_children_idxs[right_child_place],
                            child_place_in_parent=right_child_place),)
                        leaves_pair_key = (left_path[0].ast_node_idx, right_path[0].ast_node_idx)
                        all_ast_leaf_to_leaf_paths[leaves_pair_key] = tuple(itertools.chain(
                            (ASTLeaf2LeafPathNode(
                                ast_node_idx=left_path_node.ast_node_idx,
                                child_place_in_parent=left_path_node.child_place_in_parent,
                                direction=ASTLeaf2LeafPathNode.Direction.UP)
                             for left_path_node in left_path),
                            (ASTLeaf2LeafPathNode(
                                ast_node_idx=current_node_idx,
                                child_place_in_parent=None,
                                direction=ASTLeaf2LeafPathNode.Direction.COMMON),),
                            (ASTLeaf2LeafPathNode(
                                ast_node_idx=right_path_node.ast_node_idx,
                                child_place_in_parent=right_path_node.child_place_in_parent,
                                direction=ASTLeaf2LeafPathNode.Direction.DOWN)
                             for right_path_node in reversed(right_path))))
                        leaves_pair_common_ancestor[leaves_pair_key] = current_node_idx
        return [
            path + (ASTLeaf2InnerNodePathNode(ast_node_idx=child_idx, child_place_in_parent=child_place),)
            for child_place, (inner_upward_paths_from_leaves_to_child, child_idx) in
            enumerate(zip(inner_upward_paths_from_leaves_to_children,
                          current_node_children_idxs))
            for path in inner_upward_paths_from_leaves_to_child]

    all_ast_leaf_to_root_paths: List[Tuple[ASTLeaf2InnerNodePathNode, ...]] = [
        path + (ASTLeaf2InnerNodePathNode(ast_node_idx=sub_ast_root_node_idx, child_place_in_parent=None),)
        for path in aux_recursive_ast_traversal(current_node_idx=sub_ast_root_node_idx)]

    all_ast_leaf_to_root_paths: Dict[ASTNodeIdxType, Tuple[ASTLeaf2InnerNodePathNode, ...]] = {
        path[0].ast_node_idx: path for path in all_ast_leaf_to_root_paths}

    subtree_indices_range = (sub_ast_root_node_idx, leaves_sequence[-1])
    if verify_preorder_indexing:
        assert all_subtree_indices == set(range(subtree_indices_range[0], subtree_indices_range[1] + 1))

    return ASTPaths(
        leaf_to_leaf_paths=all_ast_leaf_to_leaf_paths,
        leaves_pair_common_ancestor=leaves_pair_common_ancestor,
        leaf_to_root_paths=all_ast_leaf_to_root_paths,
        leaves_sequence=tuple(leaves_sequence),
        postorder_traversal_sequence=tuple(postorder_traversal_sequence),
        siblings_sequences=siblings_sequences,
        nodes_depth=nodes_depth,
        subtree_indices_range=subtree_indices_range)


def traverse_ast(
        method_ast: SerMethodAST,
        root_sub_ast_node_idx: ASTNodeIdxType,
        subtrees_to_ignore: Optional[Set[int]] = None) -> Iterable[Tuple[SerASTNode, int, str]]:
    if subtrees_to_ignore is None:
        subtrees_to_ignore = set()
    assert root_sub_ast_node_idx not in subtrees_to_ignore
    assert root_sub_ast_node_idx < len(method_ast.nodes)
    traversal_stack = [(root_sub_ast_node_idx, 0, 'pre')]

    while len(traversal_stack) > 0:
        current_ast_node_idx, current_node_depth, pre_or_post = traversal_stack.pop()
        current_ast_node = method_ast.nodes[current_ast_node_idx]

        if pre_or_post == 'pre':
            yield current_ast_node, current_node_depth, 'pre'
            traversal_stack.append((current_ast_node_idx, current_node_depth, 'post'))
            for child_idx in reversed(current_ast_node.children_idxs):
                if child_idx not in subtrees_to_ignore:
                    traversal_stack.append((child_idx, current_node_depth + 1, 'pre'))
        else:
            assert pre_or_post == 'post'
            yield current_ast_node, current_node_depth, 'post'


def ast_node_to_str(method: SerMethod, ast_node: SerASTNode) -> str:
    props = {'identifier': ast_node.identifier,
             'name': ast_node.name,
             'modifier': ast_node.modifier,
             'type_name': ast_node.type_name}
    if len(ast_node.children_idxs) == 0 and \
            ast_node.code_sub_token_range_ref is not None and \
            ast_node.code_sub_token_range_ref.begin_token_idx == ast_node.code_sub_token_range_ref.end_token_idx and \
            method.code.tokenized[ast_node.code_sub_token_range_ref.begin_token_idx].identifier_idx is not None:
        props['token_identifier'] = method.code.tokenized[
            ast_node.code_sub_token_range_ref.begin_token_idx].identifier_idx
    props = {k: v for k, v in props.items() if v is not None}
    props_str = ' -- '.join(f'{k}={v}' for k, v in props.items())
    return f'{ast_node.type.value} [{ast_node.idx}]    {props_str}'


def print_ast(method_ast: SerMethodAST,
              method: SerMethod,
              root_sub_ast_node_idx: ASTNodeIdxType,
              subtrees_to_ignore: Optional[Set[int]] = None):
    for ast_node, depth, pop in traverse_ast(
            method_ast=method_ast,
            root_sub_ast_node_idx=root_sub_ast_node_idx,
            subtrees_to_ignore=subtrees_to_ignore):
        if pop == 'pre':
            print('  ' * depth + ast_node_to_str(method=method, ast_node=ast_node))

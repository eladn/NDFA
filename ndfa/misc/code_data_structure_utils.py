from ndfa.misc.code_data_structure_api import SerASTNodeType, SerASTNode, SerMethodAST, SerMethod, SerPDGNode
from ndfa.misc.iter_raw_extracted_data_files import RawExtractedExample

from functools import reduce
from typing import Tuple, Set


__all__ = [
    'get_pdg_node_tokenized_expression', 'find_all_simple_names_in_sub_ast', 'get_symbol_idxs_used_in_logging_call']


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
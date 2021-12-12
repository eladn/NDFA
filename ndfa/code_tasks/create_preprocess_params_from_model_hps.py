__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-10-20"

from ndfa.ndfa_model_hyper_parameters import NDFAModelHyperParams
from ndfa.code_tasks.method_code_preprocess_params import NDFAModelPreprocessParams, MethodCodePreprocessParams, \
    ASTPreprocessParams, HierarchicMethodEncoderPreprocessParams, ControlFlowPathsPreprocessParams, \
    ASTPathsPreprocessParams, NGramsPreprocessParams, ControlFlowFlatSeqPreprocessParams
from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams
from ndfa.code_nn_modules.params.cfg_single_path_macro_encoder_params import SingleFlatCFGNodesSeqMacroEncoderParams
from ndfa.nn_utils.modules.params.graph_paths_encoder_params import EdgeTypeInsertionMode
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams


__all__ = ['create_preprocess_params_from_model_hps']


def create_preprocess_params_from_ast_encoder_params(ast_encoder_params: ASTEncoderParams) -> ASTPreprocessParams:
    if ast_encoder_params.encoder_type == ast_encoder_params.EncoderType.Tree:
        return ASTPreprocessParams(dgl_tree=True)
    if ast_encoder_params.encoder_type == ast_encoder_params.EncoderType.GNN:
        return ASTPreprocessParams(pyg_graph=True)
    assert ast_encoder_params.encoder_type in \
           {ast_encoder_params.EncoderType.SetOfPaths, ast_encoder_params.EncoderType.PathsFolded}
    ast_paths_params = ASTPathsPreprocessParams(
        traversal=ast_encoder_params.paths_add_traversal_edges,
        leaf_to_leaf='leaf_to_leaf' in ast_encoder_params.ast_paths_types,
        leaf_to_leaf_shuffler=
        'leaf_to_leaf' in ast_encoder_params.ast_paths_types and ast_encoder_params.shuffle_ast_paths,
        leaf_to_root='leaf_to_root' in ast_encoder_params.ast_paths_types,
        leaf_to_root_shuffler=
        'leaf_to_root' in ast_encoder_params.ast_paths_types and ast_encoder_params.shuffle_ast_paths,
        leaves_sequence='leaves_sequence' in ast_encoder_params.ast_paths_types,
        leaves_sequence_shuffler=
        'leaves_sequence' in ast_encoder_params.ast_paths_types and ast_encoder_params.shuffle_ast_paths,
        siblings_sequences='siblings_sequences' in ast_encoder_params.ast_paths_types,
        siblings_w_parent_sequences='siblings_w_parent_sequences' in ast_encoder_params.ast_paths_types)
    return ASTPreprocessParams(paths=ast_paths_params)


def create_preprocess_params_from_model_hps(model_hps: NDFAModelHyperParams) -> NDFAModelPreprocessParams:
    pp_params = NDFAModelPreprocessParams(method_code=MethodCodePreprocessParams())
    if model_hps.method_code_encoder.method_encoder_type == model_hps.method_code_encoder.EncoderType.Hierarchic:
        hierarchic_params = model_hps.method_code_encoder.hierarchic_micro_macro_encoder
        control_flow_paths_params = None
        control_flow_single_flat_seq_params = None
        if hierarchic_params.global_context_encoder.encoder_type == \
                hierarchic_params.global_context_encoder.EncoderType.CFGPaths:
            cfg_paths_encoder = hierarchic_params.global_context_encoder.paths_encoder
            control_flow_paths_params = ControlFlowPathsPreprocessParams(
                traversal_edges=cfg_paths_encoder.edge_types_insertion_mode in
                                {EdgeTypeInsertionMode.AsStandAloneToken, EdgeTypeInsertionMode.MixWithNodeEmbedding},
                full_paths=not cfg_paths_encoder.is_ngrams,
                ngrams=NGramsPreprocessParams(
                    min_n=cfg_paths_encoder.ngrams.min_n, max_n=cfg_paths_encoder.ngrams.max_n)
                if cfg_paths_encoder.is_ngrams else None)
        elif hierarchic_params.global_context_encoder.encoder_type == \
                hierarchic_params.global_context_encoder.EncoderType.SingleFlatCFGNodesSeq:
            if hierarchic_params.global_context_encoder.single_flat_seq_encoder.cfg_nodes_order == \
                    SingleFlatCFGNodesSeqMacroEncoderParams.CFGNodesOrder.Random:
                control_flow_single_flat_seq_params = ControlFlowFlatSeqPreprocessParams(
                    cfg_nodes_random_permutation=True)
        pp_params.method_code.hierarchic = HierarchicMethodEncoderPreprocessParams(
            micro_ast=
            None if not hierarchic_params.local_expression_encoder.encoder_type ==
            CodeExpressionEncoderParams.EncoderType.AST else
            create_preprocess_params_from_ast_encoder_params(hierarchic_params.local_expression_encoder.ast_encoder),
            micro_tokens_seq=
            hierarchic_params.local_expression_encoder.encoder_type ==
            CodeExpressionEncoderParams.EncoderType.FlatTokensSeq,
            macro_ast=
            None if hierarchic_params.global_context_encoder.encoder_type !=
            hierarchic_params.global_context_encoder.EncoderType.UpperASTPaths else
            create_preprocess_params_from_ast_encoder_params(
                hierarchic_params.global_context_encoder.macro_trimmed_ast_encoder),
            control_flow_paths=control_flow_paths_params,
            control_flow_single_flat_seq=control_flow_single_flat_seq_params,
            control_flow_graph=
            hierarchic_params.global_context_encoder.encoder_type ==
            hierarchic_params.global_context_encoder.EncoderType.CFGGNN)

    elif model_hps.method_code_encoder.method_encoder_type == model_hps.method_code_encoder.EncoderType.WholeMethod:
        code_encoder = model_hps.method_code_encoder.whole_method_expression_encoder
        if code_encoder.encoder_type == CodeExpressionEncoderParams.EncoderType.FlatTokensSeq:
            pp_params.method_code.whole_method_tokens_seq = True
        elif code_encoder.encoder_type == CodeExpressionEncoderParams.EncoderType.AST:
            pp_params.method_code.whole_method_ast = create_preprocess_params_from_ast_encoder_params(
                code_encoder.ast_encoder)
        else:
            assert False
    return pp_params

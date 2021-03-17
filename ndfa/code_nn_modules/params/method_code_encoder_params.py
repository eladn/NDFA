from confclass import confclass, confparam
from ndfa.code_nn_modules.params.method_cfg_encoder_params import MethodCFGEncoderParams
from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_nn_modules.params.identifier_encoder_params import IdentifierEncoderParams


__all__ = ['MethodCodeEncoderParams']


@confclass
class MethodCodeEncoderParams:
    method_encoder_type: str = confparam(
        # default='whole-method',
        default='method-cfg-v2',
        choices=('whole-method', 'method-cfg', 'method-cfg-v2'),
        description="Representation type of the code "
                    "(main architecture of the method-code-encoder).")
    # relevant only if `method_encoder_type == 'method-cfg'`
    method_cfg_encoder: MethodCFGEncoderParams = confparam(
        default_factory=MethodCFGEncoderParams,
        description="Representation type of the method-CFG "
                    "(specific architecture of the method-CFG-code-encoder).",
        arg_prefix='method_cfg_encoder')
    # relevant only if `method_encoder_type == 'method-ast'`
    method_ast_encoder: ASTEncoderParams = confparam(
        default_factory=ASTEncoderParams,
        description="Representation type of the method-AST "
                    "(specific architecture of the method-AST-code-encoder).",
        arg_prefix='method_ast_encoder')
    # relevant only if `method_encoder_type == 'method-linear-seq'`
    method_linear_seq_expression_encoder: CodeExpressionEncoderParams = confparam(
        default_factory=CodeExpressionEncoderParams,
        description="Representation type of the whole method code as linear sequence "
                    "(part of the architecture of the code-encoder).",
        arg_prefix='method_linear_seq_encoder')

    # preprocess params
    # TODO: put the preprocess params in a dedicated nested confclass.
    max_nr_identifiers: int = confparam(
        default=110,
        description="The max number of identifiers.")
    min_nr_symbols: int = confparam(
        default=5,
        description="The max number of .")
    max_nr_symbols: int = confparam(
        default=55,
        description="The max number of .")
    max_nr_identifier_sub_parts: int = confparam(
        default=5,
        description="The max number of sub-identifiers in an identifier.")
    min_nr_tokens_method_code: int = confparam(
        default=50,
        description="The max number of .")
    max_nr_tokens_method_code: int = confparam(
        default=700,
        description="The max number of .")
    min_nr_pdg_nodes: int = confparam(
        default=6,
        description="The max number of .")
    max_nr_pdg_nodes: int = confparam(
        default=80,
        description="The max number of .")
    max_nr_tokens_in_pdg_node_expression: int = confparam(
        default=30,
        description="The max number of .")
    max_nr_pdg_edges: int = confparam(
        default=300,
        description="The max number of .")
    max_nr_pdg_data_dependency_edges_between_two_nodes: int = confparam(
        default=6,
        description="The max number of .")
    min_nr_control_flow_paths: int = confparam(
        default=1,
        description="The max number of .")
    max_nr_control_flow_paths: int = confparam(
        default=200,
        description="The max number of .")
    min_control_flow_path_len: int = confparam(
        default=3,
        description="The max number of .")
    max_control_flow_path_len: int = confparam(
        default=80,
        description="The max number of .")

    max_sub_identifier_vocab_size: int = confparam(
        default=1000,
        description="The max size of the sub-identifiers vocabulary.")

    identifier_encoder: IdentifierEncoderParams = confparam(
        default_factory=IdentifierEncoderParams,
        arg_prefix='identifier_encoder')

    symbol_embedding_dim: int = confparam(
        default=256,
        description="Embedding size for a symbol.")

    use_symbols_occurrences_for_symbols_encodings: bool = confparam(
        default=True)

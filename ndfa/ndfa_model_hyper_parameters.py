from confclass import confclass, confparam
from typing import Optional


__all__ = [
    'SequenceCombinerParams', 'SequenceEncoderParams',
    'IdentifierEncoderParams', 'ASTEncoderParams',
    'CodeExpressionEncoderParams', 'MethodCFGEncoderParams',
    'MethodCodeEncoderParams', 'TargetSymbolsDecoderParams',
    'NDFAModelHyperParams', 'NDFAModelTrainingHyperParams']


@confclass
class SequenceCombinerParams:
    method: str = confparam(
        default='attn',
        choices=('attn', 'sum', 'mean'),
        description="...")
    nr_attn_heads: int = confparam(
        default=8)
    nr_dim_reduction_layers: int = confparam(
        default=1)


@confclass
class SequenceEncoderParams:
    encoder_type: str = confparam(
        default='rnn',
        choices=('rnn', 'transformer'),
        description="...")
    rnn_type: str = confparam(
        default='lstm', choices=('lstm', 'gru'))
    nr_rnn_layers: int = confparam(
        default=1)
    bidirectional_rnn: bool = confparam(
        default=True)
    sequence_combiner: Optional[SequenceCombinerParams] = confparam(
        default=None)


@confclass
class IdentifierEncoderParams:
    identifier_embedding_dim: int = confparam(
        default=256,
        description="Embedding size for an identifier.")
    nr_sub_identifier_hashing_features: int = confparam(
        default=256)


@confclass
class ASTEncoderParams:
    encoder_type: str = confparam(
        default='set-of-paths',
        choices=('set-of-paths', 'tree'),
        description="Representation type of the AST (specific architecture of the AST code encoder).")


@confclass
class CodeExpressionEncoderParams:
    encoder_type: str = confparam(
        default='linear-seq',
        choices=('linear-seq', 'ast'),
        description="Representation type of the expression "
                    "(part of the architecture of the code-encoder).")

    # relevant only if `encoder_type == 'ast'`
    ast_encoder: ASTEncoderParams = confparam(
        default_factory=ASTEncoderParams,
        description="Representation type of the AST of the expression "
                    "(part of the architecture of the code-encoder).",
        arg_prefix='ast_encoder')

    token_type_embedding_dim: int = confparam(
        default=64,
        description="Embedding size for code token type (operator, identifier, etc).")

    kos_token_embedding_dim: int = confparam(
        default=256,
        description="Embedding size for code keyword/operator/separator token.")

    token_encoding_dim: int = confparam(
        default=256,
        # default_factory_with_self_access=lambda _self:
        # _self.identifier_embedding_size + _self.code_token_type_embedding_size,
        # default_description="identifier_embedding_size + code_token_type_embedding_size",
        description="Size of encoded code token vector.")

    combined_expression_encoding_dim: int = confparam(
        # default_as_other_field='code_token_encoding_size',
        default=512,
        description="Size of encoded combined code expression.")

    sequence_encoder: SequenceEncoderParams = confparam(
        default_factory=SequenceEncoderParams,
        arg_prefix='sequence-encoder')


@confclass
class MethodCFGEncoderParams:
    encoder_type: str = confparam(
        default='control-flow-paths-folded-to-nodes',
        choices=('set-of-control-flow-paths', 'control-flow-paths-folded-to-nodes', 'gnn',
                 'control-flow-paths-ngrams', 'set-of-nodes', 'all-nodes-single-unstructured-linear-seq',
                 'all-nodes-single-random-permutation-seq'),
        description="Representation type of the method-CFG (specific architecture of the method-CFG-code-encoder).")

    cfg_node_expression_encoder: CodeExpressionEncoderParams = confparam(
        default_factory=CodeExpressionEncoderParams,
        description="Representation type of the expression of a CFG node "
                    "(part of the architecture of the code-encoder).",
        arg_prefix='cfg_node_expression_encoder')

    cfg_node_expression_combiner: SequenceCombinerParams = confparam(
        default_factory=lambda: SequenceCombinerParams(
            method='attn', nr_attn_heads=4, nr_dim_reduction_layers=3),
        arg_prefix='cfg_node_expression_combiner')

    cfg_node_control_kinds_embedding_dim: int = confparam(
        default=64,
        description="Embedding size for the CFG node control kind.")

    cfg_node_encoding_dim: int = confparam(
        default=512,
        # default_factory_with_self_access=lambda _self:
        # _self.cfg_node_type_embedding_size + _self.code_expression_encoding_size,
        # default_description="cfg_node_type_embedding_size + code_expression_encoding_size",
        description="Size of encoded CFG node vector.")

    cfg_paths_sequence_encoder: SequenceEncoderParams = confparam(
        default_factory=SequenceEncoderParams,
        arg_prefix='sequence-encoder')


@confclass
class MethodCodeEncoderParams:
    method_encoder_type: str = confparam(
        default='method-cfg',
        choices=('method-linear-seq', 'method-ast', 'method-cfg'),
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
    method_linear_seq_expression_encoder_type: CodeExpressionEncoderParams = confparam(
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
        default_factory=IdentifierEncoderParams)

    symbol_embedding_dim: int = confparam(
        default=256,
        description="Embedding size for a symbol.")

    use_symbols_occurrences_for_symbols_encodings: bool = confparam(
        default=True)


@confclass
class TargetSymbolsDecoderParams:
    # logging call task params:
    # TODO: move to logging-call task specific params
    min_nr_target_symbols: int = confparam(
        default=1,
        description="The max number of .")
    max_nr_target_symbols: int = confparam(
        default=4,
        description="The max number of .")
    use_batch_flattened_target_symbols_vocab: bool = confparam(
        default=False)


@confclass
class NDFAModelHyperParams:
    activation_fn: str = confparam(
        default='relu',
        choices=('relu', 'prelu', 'leaky_relu', 'sigmoid', 'tanh', 'none'),
        description='Activation function type to use for non-linearities all over the model.')
    method_code_encoder: MethodCodeEncoderParams = confparam(
        default_factory=MethodCodeEncoderParams,
        arg_prefix='code-encoder')
    target_symbols_decoder: TargetSymbolsDecoderParams = confparam(
        default_factory=TargetSymbolsDecoderParams,
        description='...',
        arg_prefix='tgt-symbols-decoder')


@confclass
class NDFAModelTrainingHyperParams:
    dropout_rate: float = confparam(
        default=0.3,
        description="Dropout rate used during training.")

    optimizer: str = confparam(
        choices=('adam', 'wadam'),
        default='wadam')

    eff_batch_size: int = confparam(
        default=64,
        description="Batch size both for training (must be a multiplication of the used batch size).")

    nr_epochs: int = confparam(
        default=2000,
        description="Number of epochs to train.")

    stop_criterion: Optional[str] = confparam(
        choices=('early-stopping',),
        default=None,
        description="Criterion for stopping training.")

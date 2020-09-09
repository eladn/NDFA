from confclass import confclass, confparam
from typing import Optional


__all__ = ['MethodCodeEncoderParams', 'NDFAModelHyperParams', 'NDFAModelTrainingHyperParams']


@confclass
class MethodCodeEncoderParams:
    encoder_type: str = confparam(
        default='cfg-paths',
        choices=('linear-seq', 'ast-paths', 'ast', 'cfg-paths', 'cfg'),
        description="Representation type of the code (architecture of the code-encoder).")
    pdg_expression_encoder_type: str = confparam(
        default='method-linear-seq',
        choices=('linear-seq', 'method-linear-seq', 'ast-paths', 'ast'),
        description="Representation type of the expression of a PDG node "
                    "(part of the architecture of the code-encoder).")

    # preprocess params
    # TODO: put the preprocess params in a dedicated nested confclass.
    max_nr_identifiers: int = confparam(
        default=110,
        description="The max number of sub-identifiers in an identifier.")
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
    min_nr_pdg_paths: int = confparam(
        default=6,
        description="The max number of .")
    max_nr_pdg_paths: int = confparam(
        default=400,
        description="The max number of .")

    # logging call task params:
    # TODO: move to logging-call task specific params
    min_nr_target_symbols: int = confparam(
        default=1,
        description="The max number of .")
    max_nr_target_symbols: int = confparam(
        default=4,
        description="The max number of .")

    max_sub_identifier_vocab_size: int = confparam(
        default=500,
        description="The max size of the sub-identifiers vocabulary.")

    identifier_embedding_size: int = confparam(
        default=128,
        description="Embedding size for an identifier.")

    code_token_type_embedding_size: int = confparam(
        default=8,
        description="Embedding size for code token type (operator, identifier, etc).")

    code_token_encoding_size: int = confparam(
        default_factory_with_self_access=lambda _self:
        _self.identifier_embedding_size + _self.code_token_type_embedding_size,
        default_description="identifier_embedding_size + code_token_type_embedding_size",
        description="Size of encoded code token vector.")

    code_expression_encoding_size: int = confparam(
        default_as_other_field='code_token_encoding_size',
        description="Size of encoded code expression vector.")

    cfg_node_type_embedding_size: int = confparam(
        default=4,
        description="Embedding size for the CFG node type.")

    cfg_node_encoding_size: int = confparam(
        default_factory_with_self_access=lambda _self:
        _self.cfg_node_type_embedding_size + _self.code_expression_encoding_size,
        default_description="cfg_node_type_embedding_size + code_expression_encoding_size",
        description="Size of encoded CFG node vector.")


@confclass
class NDFAModelHyperParams:
    method_code_encoder: MethodCodeEncoderParams = confparam(
        default_factory=MethodCodeEncoderParams,
        arg_prefix='code-encoder')

    use_batch_flattened_target_symbols_vocab: bool = confparam(
        default=False)


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

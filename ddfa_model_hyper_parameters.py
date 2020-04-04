from confclass import confclass, confparam
from typing import Optional


__all__ = ['DDFAModelHyperParams', 'DDFAModelTrainingHyperParams']


@confclass
class DDFAModelHyperParams:
    max_identifier_sub_parts: int = confparam(
        default=7,
        description="The max number of sub-identifiers in an identifier.")

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
class DDFAModelTrainingHyperParams:
    dropout_keep_rate: float = confparam(
        default=0.75,
        description="Dropout rate used during training.")

    optimizer: str = confparam(
        choices=('adam', 'sgd'),
        default='adam')

    batch_size: int = confparam(
        default=64,
        description="Batch size both for training and for evaluating.")

    nr_epochs: int = confparam(
        default=20,
        description="Number of epochs to train.")

    stop_criterion: Optional[str] = confparam(
        choices=(),  # TODO: put here some stop criterions
        default=None,
        description="Number of epochs to train.")

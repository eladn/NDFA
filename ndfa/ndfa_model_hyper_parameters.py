from dataclasses import dataclass
from typing import Optional

from ndfa.code_nn_modules.params.method_code_encoder_params import MethodCodeEncoderParams
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams
from ndfa.misc.configurations_utils import conf_field, DeterministicallyHashable


__all__ = [
    'TargetSymbolsDecoderParams',
    'NDFAModelHyperParams', 'NDFAModelTrainingHyperParams']


@dataclass
class TargetSymbolsDecoderParams:
    # logging call task params:
    # TODO: move to logging-call task specific params
    min_nr_target_symbols: int = conf_field(
        default=1,
        description="The max number of .")
    max_nr_target_symbols: int = conf_field(
        default=4,
        description="The max number of .")
    use_batch_flattened_target_symbols_vocab: bool = conf_field(
        default=False)


@dataclass
class NDFAModelHyperParams(DeterministicallyHashable):
    activation_fn: str = conf_field(
        default='leaky_relu',
        choices=('relu', 'prelu', 'leaky_relu', 'sigmoid', 'tanh', 'none'),
        description='Activation function type to use for non-linearities all over the model.')
    normalization: NormWrapperParams = conf_field(
        default_factory=lambda: NormWrapperParams(norm_type=NormWrapperParams.NormType.Layer))
    method_code_encoder: MethodCodeEncoderParams = conf_field(
        default_factory=MethodCodeEncoderParams,
        arg_prefix='code-encoder')
    target_symbols_decoder: TargetSymbolsDecoderParams = conf_field(
        default_factory=TargetSymbolsDecoderParams,
        description='...',
        arg_prefix='tgt-symbols-decoder')


@dataclass
class NDFAModelTrainingHyperParams:
    dropout_rate: float = conf_field(
        default=0.3,
        description="Dropout rate used during training.")

    optimizer: str = conf_field(
        choices=('adam', 'wadam'),
        default='wadam')

    eff_batch_size: int = conf_field(
        default=64,
        description="Batch size both for training (must be a multiplication of the used batch size).")

    nr_epochs: int = conf_field(
        default=2000,
        description="Number of epochs to train.")

    stop_criterion: Optional[str] = conf_field(
        choices=('early-stopping',),
        default=None,
        description="Criterion for stopping training.")

    gradient_clip: Optional[float] = conf_field(
        default=0.5)

    weight_decay: Optional[float] = conf_field(
        default=1e-2)

    learning_rate: Optional[float] = conf_field(
        default=0.0003)

    learning_rate_decay: Optional[float] = conf_field(
        default=0.01)

    reduce_lr_on_plateau: bool = conf_field(
        default=True)

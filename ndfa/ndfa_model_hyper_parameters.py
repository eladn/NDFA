from confclass import confparam
from dataclasses import dataclass
from typing import Optional

from ndfa.code_nn_modules.params.method_code_encoder_params import MethodCodeEncoderParams


__all__ = [
    'TargetSymbolsDecoderParams',
    'NDFAModelHyperParams', 'NDFAModelTrainingHyperParams']


@dataclass
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


@dataclass
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


@dataclass
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

    gradient_clip: Optional[float] = confparam(
        default=0.5)

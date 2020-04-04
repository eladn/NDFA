from confclass import confclass, confparam
from typing import Optional
import argparse

from experiment_setting import ExperimentSetting


@confclass
class ModelExecutionParams:
    model_save_path: Optional[str] = confparam(
        default=None,
        description="Path to save the model file into.",
        arg_names=['--model-save-path'])

    model_load_path: Optional[str] = confparam(
        default=None,
        description="Path to load the model from.",
        arg_names=['--model-load-path'])

    train_data_path: Optional[str] = confparam(
        default=None,
        description="Path to preprocessed dataset.",
        arg_names=['--train-data-path'])

    eval_data_path: Optional[str] = confparam(
        default=None,
        description="Path to preprocessed evaluation data.",
        arg_names=['--eval-data-path'])

    predict_data_path: Optional[str] = confparam(
        default=None,
        description="Path to preprocessed prediction data.",
        arg_names=['--pred-data-path'])

    verbose_mode: int = confparam(
        default=1,
        choices=(0, 1, 2),
        description="Verbose mode (should be in {0,1,2}).",
        arg_names=['--verbosity', '-v'])

    logs_path: Optional[str] = confparam(
        default=None,
        description="Path to store logs into. if not given logs are not saved to file.",
        arg_names=['--logs-path', '-lp'])

    use_tensorboard: bool = confparam(
        default=False,
        description="Use tensorboard during training.",
        arg_names=['--use-tensorboard'])

    num_train_epochs: int = confparam(
        default=20,
        description="The max number of epochs to train the model. Stopping earlier must be done manually (kill).")

    save_every_epochs: int = confparam(
        default=1,
        description="After how many training iterations a model should be saved.")

    num_batches_to_log_progress: int = confparam(
        default=100,
        description="Number of batches (during training / evaluating) to complete between two progress-logging "
                    "records.")

    num_train_batches_to_evaluate: int = confparam(
        default=100,
        description="Number of training batches to complete between model evaluations on the test set.")

    max_latest_checkpoints_to_keep: int = confparam(
        default=10,
        description="Keep this number of newest trained versions during training.")

    experiment_setting: ExperimentSetting = confparam(
        default_factory=ExperimentSetting,
        description="Experiment setting.",
        arg_prefix='expr'
    )

    use_gpu_if_available: bool = confparam(
        default=True,
        description="Use GPU if available.",
        arg_names=['--use-gpu']
    )

    @property
    def perform_training(self) -> bool:
        return bool(self.train_data_path)

    @property
    def perform_evaluation(self) -> bool:
        return bool(self.eval_data_path)

    @property
    def perform_prediction(self):
        return bool(self.predict_data_path)

    @property
    def should_load_model(self) -> bool:
        return bool(self.model_load_path)

    @property
    def should_save_model(self) -> bool:
        return bool(self.model_save_path)

    # def train_steps_per_epoch(self, num_train_examples: int) -> int:
    #     return common.nr_steps(num_train_examples, self.train_batch_size)
    #
    # def test_steps(self, num_test_examples: int) -> int:
    #     return common.nr_steps(num_test_examples, self.test_batch_size)

    def data_path(self, is_evaluating: bool = False):
        return self.test_data_path if is_evaluating else self.train_data_path

    # @property
    # def word_freq_dict_path(self) -> Optional[str]:
    #     if not self.perform_training:
    #         return None
    #     return f'{self.train_data_path_prefix}.dict.c2v'
    #
    # @classmethod
    # def get_vocabularies_path_from_model_path(cls, model_file_path: str) -> str:
    #     vocabularies_save_file_name = "dictionaries.bin"
    #     return '/'.join(model_file_path.rstrip('/').split('/')[:-1] + [vocabularies_save_file_name])

    def __verify_conf__(self):
        if not self.perform_training and not self.should_load_model:
            raise argparse.ArgumentError(None, "Must train or load a model.")


def test_model_execution_params():
    model_execution_params: ModelExecutionParams = ModelExecutionParams.factory(
        load_from_args=True, load_from_yaml=True, verify_confclass=True)
    print(f'params.model_hyper_params.code_vector_size: {model_execution_params.model_hyper_params.code_vector_size}')
    print(model_execution_params)
    model_execution_params.pprint()
    model_execution_params.save_to_yaml(export_only_explicitly_set_params=True)
    print(f'__explicitly_set_params__: {model_execution_params.__explicitly_set_params__}')


if __name__ == '__main__':
    test_model_execution_params()

from dataclasses import dataclass
from typing import Optional
import argparse

from ndfa.experiment_setting import ExperimentSetting
from ndfa.misc.configurations_utils import conf_field


__all__ = ['ModelExecutionParams']


@dataclass
class ModelExecutionParams:
    model_save_path: Optional[str] = conf_field(
        default=None,
        description="Path to save the model file into during and after training.",
        arg_names=['--model-save-path'])

    model_load_path: Optional[str] = conf_field(
        default=None,
        description="Path to load the model from.",
        arg_names=['--model-load-path'])

    pp_data_dir_path: Optional[str] = conf_field(
        default=None,
        description="Path to preprocessed dataset.",
        arg_names=['--pp-data'])

    predict_pp_data_path: Optional[str] = conf_field(
        default=None,
        description="Path to preprocessed prediction data.",
        arg_names=['--pred-pp-data-path'])

    predict_raw_data_path: Optional[str] = conf_field(
        default=None,
        description="Path to raw prediction data.",
        arg_names=['--pred-raw-data-path'])

    predict_output_path: Optional[str] = conf_field(
        default=None,
        description="Dir path to prediction outputs.",
        arg_names=['--pred-output-path'])

    perform_training: bool = conf_field(
        default=False,
        description="Train of the model.",
        arg_names=['--train'])

    perform_evaluation: bool = conf_field(
        default=False,
        description="Evaluate of the model. If `--train` has also been set, evaluate during and after the training.",
        arg_names=['--eval'])

    perform_preprocessing: bool = conf_field(
        default=False,
        description="Perform preprocessing of the raw dataset.",
        arg_names=['--preprocess'])

    get_pp_data_params_hash: bool = conf_field(
        default=False,
        description="Get the hash of the required preprocessed data params matching the given model hyper params.")

    pp_nr_processes: Optional[int] = conf_field(
        default=4,
        description="Number of processes to use for preprocessing.",
        arg_names=['--pp-nr-processes'])

    pp_override: bool = conf_field(
        default=False,
        description="Override existing preprocessed data if such exist in the given `pp_data` destination path.",
        arg_names=['--pp-override'])

    pp_storage_method: str = conf_field(
        default='rocksdb',
        choices=('zip', 'tar', 'dbm', 'rocksdb'),
        description="Override existing preprocessed data if such exist in the given `pp_data` destination path.",
        arg_names=['--pp-storage-method'])

    pp_compression_method: str = conf_field(
        default='none',
        choices=('none', 'gzip', 'bz2', 'lzma', 'snappy', 'lz4hc', 'xpress', 'zstd', 'zstdnotfinal'),
        description="Override existing preprocessed data if such exist in the given `pp_data` destination path.",
        arg_names=['--pp-compress'])

    raw_train_data_path: Optional[str] = conf_field(
        default=None,
        description="Path to raw train dataset.",
        arg_names=['--raw-train-data-path'])

    raw_validation_data_path: Optional[str] = conf_field(
        default=None,
        description="Path to raw evaluation dataset.",
        arg_names=['--raw-validation-data-path'])

    raw_test_data_path: Optional[str] = conf_field(
        default=None,
        description="Path to raw test dataset.",
        arg_names=['--raw-test-data-path'])

    verbose_mode: int = conf_field(
        default=1,
        choices=(0, 1, 2),
        description="Verbose mode (should be in {0,1,2}).",
        arg_names=['--verbosity', '-v'])

    logs_path: Optional[str] = conf_field(
        default=None,
        description="Path to store logs into. if not given logs are not saved to file.",
        arg_names=['--logs-path', '-lp'])

    use_tensorboard: bool = conf_field(
        default=False,
        description="Use tensorboard during training.",
        arg_names=['--use-tensorboard'])

    use_gdrive_logger: bool = conf_field(
        default=False,
        description="Use google drive logger during training.",
        arg_names=['--use-gdrive-logger'])

    train_results_gdrive_folder_id: Optional[str] = conf_field(
        default=None,
        description="Google drive folder ID for storing training logs.")

    num_train_epochs: int = conf_field(
        default=20,
        description="The max number of epochs to train the model. Stopping earlier must be done manually (kill).")

    save_every_epochs: int = conf_field(
        default=1,
        description="After how many training iterations a model should be saved.")

    num_batches_to_log_progress: int = conf_field(
        default=100,
        description="Number of batches (during training / evaluating) to complete between two progress-logging "
                    "records.")

    num_train_batches_to_evaluate: int = conf_field(
        default=100,
        description="Number of training batches to complete between model evaluations on the test set.")

    max_latest_checkpoints_to_keep: Optional[int] = conf_field(
        default=10,
        description="Keep this number of newest trained versions during training.")

    experiment_setting: ExperimentSetting = conf_field(
        default_factory=ExperimentSetting,
        description="Experiment setting.",
        arg_prefix='expr'
    )

    batch_size: int = conf_field(
        default=16,
        description="Actual batch size both for training and for evaluating "
                    "(must be a divisor of `experiment_setting.train.eff_batch_size` for training).")

    use_gpu_if_available: bool = conf_field(
        default=True,
        description="Use GPU if available.",
        arg_names=['--use-gpu']
    )

    seed: Optional[int] = conf_field(
        default=1)

    use_notify: bool = conf_field(
        default=False,
        arg_names=['--use-notify'])

    dataloader_num_workers: Optional[int] = conf_field(
        default=None)

    dataloader_pin_memory: bool = conf_field(
        default=True)

    progress_bar_min_interval_sec: float = conf_field(
        default=0.1)

    @property
    def perform_prediction(self):
        return bool(self.predict_pp_data_path) or bool(self.predict_raw_data_path)

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
        if not any((self.perform_preprocessing, self.perform_training,
                    self.perform_evaluation, self.perform_prediction)):
            raise argparse.ArgumentError(None, "Please choose one of {--preprocess, --train, --eval, --predict}.")
        if self.perform_preprocessing and (self.perform_training or self.perform_evaluation or self.perform_prediction):
            raise argparse.ArgumentError(
                None, "Cannot perform both preprocessing and training/evaluation/prediction in the same execution.")
        if self.perform_prediction and self.perform_training:
            raise argparse.ArgumentError(None, "Cannot perform both prediction and training in the same execution.")
        if self.perform_evaluation and not self.perform_training and not self.should_load_model:
            raise argparse.ArgumentError(None, "Must train or load a model in order to perform evaluation.")
        if self.perform_prediction and not self.should_load_model:
            raise argparse.ArgumentError(None, "Must load a model in order to perform prediction.")
        if bool(self.predict_raw_data_path) and bool(self.predict_pp_data_path):
            raise argparse.ArgumentError(None, "Cannot specify both `--pred-raw-data-path` and `--pred-pp-data-path`.")
        if not self.perform_training and self.should_save_model:
            raise argparse.ArgumentError(None, "Must train model in order to save the model.")
        if self.perform_training and not self.should_save_model:
            raise argparse.ArgumentError(None, "Must specify model save path if performing model training.")
        if self.perform_preprocessing and not (self.raw_train_data_path or self.raw_validation_data_path or self.raw_test_data_path):
            raise argparse.ArgumentError(None, "Must specify raw train/eval/test data path if performing preprocessing.")
        if not self.perform_preprocessing and (self.raw_train_data_path or self.raw_validation_data_path or self.raw_test_data_path):
            raise argparse.ArgumentError(None, "Must specify `--preprocess` if specifying raw data path.")
        if (self.perform_preprocessing or self.perform_training or self.perform_evaluation) and not self.pp_data_dir_path:
            raise argparse.ArgumentError(None, "Must specify `--pp-data` for performing preprocess / train / evaluation.")
        if self.experiment_setting.train_hyper_params.eff_batch_size < self.batch_size or \
                self.experiment_setting.train_hyper_params.eff_batch_size % self.batch_size != 0:
            raise argparse.ArgumentError(None, "The effective train batch size must be greater than and a "
                                               "multiplication of the actual batch size.")

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

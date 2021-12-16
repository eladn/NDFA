__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-10-23"

import time
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

from ndfa.nn_utils.model_wrapper.train_callback import TrainCallback
from ndfa.nn_utils.model_wrapper.window_average import WindowAverage
from ndfa.nn_utils.model_wrapper.gdrive_train_logger import GDriveTrainLogger


__all__ = ['GDriveTrainLoggerCallback']


class GDriveTrainLoggerCallback(TrainCallback):
    def __init__(self, gdrive_train_logger: GDriveTrainLogger, partial_epoch_results_every_mins: Optional[int] = 5):
        self.gdrive_train_logger = gdrive_train_logger
        self.full_epochs_results = {}
        self.partial_epochs_results = {}
        self.partial_epochs_eval_results = {}
        self.partial_epoch_results_last_msg_time = None
        self.partial_epoch_results_every_mins = partial_epoch_results_every_mins
        self.epoch_last_taken_start_time = None
        self.eval_start_time = None
        self.cur_epoch_time = None
        self.directories_to_backup_on_epoch_end: List[Path] = []

    def epoch_start(self, epoch_nr: int, step_nr: int, learning_rates: Tuple[float, ...]):
        self.cur_epoch_time = datetime.timedelta()
        self.epoch_last_taken_start_time = datetime.datetime.now()

    def step_start(
            self, epoch_nr: int, step_nr: int, batch_nr: int, nr_batches_in_epoch: int,
            avg_throughput: float, learning_rates: Tuple[float, ...]):
        if self.epoch_last_taken_start_time is None:
            self.epoch_last_taken_start_time = datetime.datetime.now()

    def step_end(
            self, epoch_nr: int, step_nr: int, batch_nr: int, nr_batches_in_epoch: int,
            batch_loss: float, batch_nr_examples: int, epoch_avg_loss: float,
            epoch_moving_win_loss: WindowAverage, avg_throughput: float,
            learning_rates: Tuple[float, ...]):
        if self.partial_epoch_results_every_mins is None:
            return
        if self.epoch_last_taken_start_time is not None:
            self.cur_epoch_time += (datetime.datetime.now() - self.epoch_last_taken_start_time)
            self.epoch_last_taken_start_time = None
        if self.partial_epoch_results_last_msg_time is None:
            self.partial_epoch_results_last_msg_time = time.time()
        elif self.partial_epoch_results_last_msg_time + self.partial_epoch_results_every_mins * 60 <= time.time():
            partial_epoch_results = {
                'epoch_nr': epoch_nr,
                'step_nr': step_nr,
                'batch_nr': batch_nr,
                'nr_batches_in_epoch': nr_batches_in_epoch,
                'epoch_avg_loss': epoch_avg_loss,
                'epoch_moving_win_loss': epoch_moving_win_loss.get_window_avg(),
                'avg_throughput': avg_throughput,
                'epoch_time': self.cur_epoch_time.total_seconds() / 60}
            self.partial_epochs_results[step_nr] = partial_epoch_results
            self.gdrive_train_logger.upload_as_json_file(
                self.partial_epochs_results, 'partial_epochs_results.json')
            self.partial_epoch_results_last_msg_time = time.time()

    def epoch_end_before_evaluation(
            self, epoch_nr: int, step_nr: int, epoch_avg_loss: float,
            epoch_moving_win_loss: WindowAverage, avg_throughput: float,
            learning_rates: Tuple[float, ...]):
        if self.epoch_last_taken_start_time is not None:
            self.cur_epoch_time += (datetime.datetime.now() - self.epoch_last_taken_start_time)
            self.epoch_last_taken_start_time = None
        self.eval_start_time = datetime.datetime.now()

    def epoch_end_after_evaluation(
            self, epoch_nr: int, step_nr: int, epoch_avg_loss: float,
            epoch_moving_win_loss: WindowAverage, validation_loss: float,
            validation_metrics_results: Dict[str, float], avg_throughput: float,
            learning_rates: Tuple[float, ...]):
        eval_time = (datetime.datetime.now() - self.eval_start_time).total_seconds() / 60
        self.eval_start_time = None
        epoch_results = {
            'epoch_nr': epoch_nr,
            'step_nr': step_nr,
            'epoch_avg_loss': epoch_avg_loss,
            'epoch_moving_win_loss': epoch_moving_win_loss.get_window_avg(),
            'validation_loss': validation_loss,
            'validation_metrics': validation_metrics_results,
            'avg_throughput': avg_throughput,
            'epoch_time': self.cur_epoch_time.total_seconds() / 60,
            'eval_time': eval_time}
        self.full_epochs_results[epoch_nr] = epoch_results
        self.gdrive_train_logger.upload_as_json_file(
            self.full_epochs_results, 'full_epochs_results.json')

    def step_end_before_evaluation(
            self, epoch_nr: int, step_nr: int, batch_nr: int, nr_batches_in_epoch: int,
            batch_loss: float, batch_nr_examples: int, epoch_avg_loss: float,
            epoch_moving_win_loss: WindowAverage, avg_throughput: float,
            learning_rates: Tuple[float, ...]):
        if self.epoch_last_taken_start_time is not None:
            self.cur_epoch_time += (datetime.datetime.now() - self.epoch_last_taken_start_time)
            self.epoch_last_taken_start_time = None
        self.eval_start_time = datetime.datetime.now()

    def step_end_after_evaluation(
            self, epoch_nr: int, step_nr: int, batch_nr: int, nr_batches_in_epoch: int, batch_loss: float,
            batch_nr_examples: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage,
            validation_loss: float, validation_metrics_results: Dict[str, float],
            avg_throughput: float, learning_rates: Tuple[float, ...]):
        eval_time = (datetime.datetime.now() - self.eval_start_time).total_seconds() / 60
        self.eval_start_time = None
        epoch_results = {
            'epoch_nr': epoch_nr,
            'step_nr': step_nr,
            'batch_nr': batch_nr,
            'nr_batches_in_epoch': nr_batches_in_epoch,
            'epoch_avg_loss': epoch_avg_loss,
            'epoch_moving_win_loss': epoch_moving_win_loss.get_window_avg(),
            'validation_loss': validation_loss,
            'validation_metrics': validation_metrics_results,
            'avg_throughput': avg_throughput,
            'epoch_time': self.cur_epoch_time.total_seconds() / 60,
            'eval_time': eval_time}
        self.partial_epochs_eval_results[step_nr] = epoch_results
        self.gdrive_train_logger.upload_as_json_file(
            self.partial_epochs_eval_results, 'partial_epochs_eval_results.json')

    def epoch_end(
            self, epoch_nr: int, step_nr: int, epoch_avg_loss: float,
            epoch_moving_win_loss: WindowAverage, avg_throughput: float,
            learning_rates: Tuple[float, ...]):
        self.cur_epoch_time = None
        self.epoch_last_taken_start_time = None
        self.eval_start_time = None
        self.partial_epoch_results_last_msg_time = None
        for dir_path in self.directories_to_backup_on_epoch_end:
            self.gdrive_train_logger.upload_dir(dir_path)

    def register_dir_backup_on_epoch_end(self, dir_path: Union[str, Path]):
        dir_path = dir_path if isinstance(dir_path, Path) else Path(dir_path)
        self.directories_to_backup_on_epoch_end.append(dir_path)

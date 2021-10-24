import time
import datetime
from typing import Dict, Optional
from collections import defaultdict

from ndfa.nn_utils.model_wrapper.train_callback import TrainCallback
from ndfa.nn_utils.model_wrapper.window_average import WindowAverage
from ndfa.nn_utils.model_wrapper.gdrive_train_logger import GDriveTrainLogger


__all__ = ['GDriveTrainLoggerCallback']


class GDriveTrainLoggerCallback(TrainCallback):
    def __init__(self, gdrive_train_logger: GDriveTrainLogger, partial_epoch_results_every_mins: Optional[int] = 5):
        self.gdrive_train_logger = gdrive_train_logger
        self.full_epochs_results = {}
        self.partial_epochs_results = defaultdict(dict)
        self.partial_epochs_eval_results = defaultdict(dict)
        self.partial_epoch_results_last_msg_time = None
        self.partial_epoch_results_every_mins = partial_epoch_results_every_mins
        self.epoch_last_taken_start_time = None
        self.eval_start_time = None
        self.cur_epoch_time = None

    def epoch_start(self, epoch_nr: int):
        self.cur_epoch_time = datetime.timedelta()
        self.epoch_last_taken_start_time = datetime.datetime.now()

    def step_start(
            self, epoch_nr: int, step_nr: int, nr_steps: int, avg_throughput: float):
        if self.epoch_last_taken_start_time is None:
            self.epoch_last_taken_start_time = datetime.datetime.now()

    def step_end(
            self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float, batch_nr_examples: int,
            epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, avg_throughput: float):
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
                'nr_steps': nr_steps,
                'epoch_avg_loss': epoch_avg_loss,
                'epoch_moving_win_loss': epoch_moving_win_loss.get_window_avg(),
                'avg_throughput': avg_throughput,
                'epoch_time': self.cur_epoch_time.min}
            self.partial_epochs_results[epoch_nr][step_nr] = partial_epoch_results
            self.gdrive_train_logger.upload_as_json_file(
                self.partial_epochs_results, 'partial_epochs_results.json')
            self.partial_epoch_results_last_msg_time = time.time()

    def epoch_end_before_evaluation(
            self, epoch_nr: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, avg_throughput: float):
        if self.epoch_last_taken_start_time is not None:
            self.cur_epoch_time += (datetime.datetime.now() - self.epoch_last_taken_start_time)
            self.epoch_last_taken_start_time = None
        self.eval_start_time = datetime.datetime.now()

    def epoch_end_after_evaluation(
            self, epoch_nr: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage,
            validation_loss: float, validation_metrics_results: Dict[str, float], avg_throughput: float):
        eval_time = (datetime.datetime.now() - self.eval_start_time).min
        self.eval_start_time = None
        epoch_results = {
            'epoch_nr': epoch_nr,
            'epoch_avg_loss': epoch_avg_loss,
            'epoch_moving_win_loss': epoch_moving_win_loss.get_window_avg(),
            'validation_loss': validation_loss,
            'validation_metrics': validation_metrics_results,
            'avg_throughput': avg_throughput,
            'epoch_time': self.cur_epoch_time.min,
            'eval_time': eval_time}
        self.full_epochs_results[epoch_nr] = epoch_results
        self.gdrive_train_logger.upload_as_json_file(
            self.full_epochs_results, 'full_epochs_results.json')

    def step_end_before_evaluation(
            self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float, batch_nr_examples: int,
            epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, avg_throughput: float):
        if self.epoch_last_taken_start_time is not None:
            self.cur_epoch_time += (datetime.datetime.now() - self.epoch_last_taken_start_time)
            self.epoch_last_taken_start_time = None
        self.eval_start_time = datetime.datetime.now()

    def step_end_after_evaluation(
            self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float, batch_nr_examples: int,
            epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, validation_loss: float,
            validation_metrics_results: Dict[str, float], avg_throughput: float):
        eval_time = (datetime.datetime.now() - self.eval_start_time).min
        self.eval_start_time = None
        epoch_results = {
            'epoch_nr': epoch_nr,
            'step_nr': step_nr,
            'nr_steps': nr_steps,
            'epoch_avg_loss': epoch_avg_loss,
            'epoch_moving_win_loss': epoch_moving_win_loss.get_window_avg(),
            'validation_loss': validation_loss,
            'validation_metrics': validation_metrics_results,
            'avg_throughput': avg_throughput,
            'epoch_time': self.cur_epoch_time.min,
            'eval_time': eval_time}
        self.partial_epochs_eval_results[epoch_nr][step_nr] = epoch_results
        self.gdrive_train_logger.upload_as_json_file(
            self.partial_epochs_eval_results, 'partial_epochs_eval_results.json')

    def epoch_end(
            self, epoch_nr: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, avg_throughput: float):
        self.cur_epoch_time = None
        self.epoch_last_taken_start_time = None
        self.eval_start_time = None

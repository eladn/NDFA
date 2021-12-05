__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-12-05"

import time
import datetime
import dataclasses
from typing import Dict, Optional, Any, Iterable, Tuple

import wandb

from ndfa.nn_utils.model_wrapper.train_callback import TrainCallback
from ndfa.nn_utils.model_wrapper.window_average import WindowAverage


__all__ = ['WAndBCallback']


# TODO: move to misc / utils aux file
def _flatten_structured_object(obj: object, fields_delimiter: str = '.') -> Dict[str, Any]:
    ret_list = {}

    def _structured_obj_fields_iterator(_obj: object) -> Iterable[Tuple[str, Any]]:
        if dataclasses.is_dataclass(_obj):
            return ((field.name, getattr(_obj, field.name)) for field in dataclasses.fields(_obj))
        elif isinstance(_obj, dict):
            return _obj.items()
        else:
            assert False

    def _is_obj_structured(_obj: object) -> bool:
        return dataclasses.is_dataclass(_obj) or isinstance(_obj, dict)

    def _aux_object_traversal(_obj: object, prefix: str = ''):
        for field_name, field_value in _structured_obj_fields_iterator(_obj):
            nested_prefix = f"{prefix}{fields_delimiter if prefix else ''}{field_name}"
            if _is_obj_structured(field_value):
                _aux_object_traversal(_obj=field_value, prefix=nested_prefix)
            else:
                assert nested_prefix not in ret_list
                ret_list[nested_prefix] = field_value

    _aux_object_traversal(_obj=obj)
    return ret_list


class WAndBCallback(TrainCallback):
    def __init__(self, partial_epoch_results_every_sec: Optional[int] = 15):
        self.partial_epoch_results_last_msg_time = None
        self.partial_epoch_results_every_sec = partial_epoch_results_every_sec
        self.last_taken_step_start_time_within_epoch = None
        self.eval_start_time = None
        self.cur_epoch_time = None

    def epoch_start(self, epoch_nr: int):
        self.cur_epoch_time = datetime.timedelta()
        self.last_taken_step_start_time_within_epoch = datetime.datetime.now()

    def step_start(
            self, epoch_nr: int, step_nr: int, nr_steps: int, avg_throughput: float):
        if self.last_taken_step_start_time_within_epoch is None:
            self.last_taken_step_start_time_within_epoch = datetime.datetime.now()

    def step_end(
            self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float, batch_nr_examples: int,
            epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, avg_throughput: float):
        if self.partial_epoch_results_every_sec is not None:
            if self.last_taken_step_start_time_within_epoch is not None:
                self.cur_epoch_time += (datetime.datetime.now() - self.last_taken_step_start_time_within_epoch)
                self.last_taken_step_start_time_within_epoch = None
            if self.partial_epoch_results_last_msg_time is None:
                self.partial_epoch_results_last_msg_time = time.time()
            if self.partial_epoch_results_last_msg_time + self.partial_epoch_results_every_sec > time.time():
                return
        partial_epoch_results = {
            'train/epoch': epoch_nr,
            'train/batch': step_nr,
            'train/epoch_avg_loss': epoch_avg_loss,
            'train/moving_win_loss': epoch_moving_win_loss.get_window_avg(),
            'train/avg_throughput': avg_throughput,
            'train/epoch_duration': self.cur_epoch_time.total_seconds() / 60}
        wandb.log(partial_epoch_results)
        self.partial_epoch_results_last_msg_time = time.time()

    def epoch_end_before_evaluation(
            self, epoch_nr: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, avg_throughput: float):
        if self.last_taken_step_start_time_within_epoch is not None:
            self.cur_epoch_time += (datetime.datetime.now() - self.last_taken_step_start_time_within_epoch)
            self.last_taken_step_start_time_within_epoch = None
        self.eval_start_time = datetime.datetime.now()

    def epoch_end_after_evaluation(
            self, epoch_nr: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage,
            validation_loss: float, validation_metrics_results: Dict[str, float], avg_throughput: float):
        eval_time = (datetime.datetime.now() - self.eval_start_time).total_seconds() / 60
        self.eval_start_time = None
        epoch_results = {
            'train/epoch': epoch_nr,
            'train/epoch_avg_loss': epoch_avg_loss,
            'train/moving_win_loss': epoch_moving_win_loss.get_window_avg(),
            'train/avg_throughput': avg_throughput,
            'train/epoch_duration': self.cur_epoch_time.total_seconds() / 60,
            'val/loss': validation_loss,
            'val/duration': eval_time,
            **{f'val/metrics/{k}': v
               for k, v in _flatten_structured_object(validation_metrics_results, fields_delimiter='/').items()}
        }
        wandb.log(epoch_results)

    def step_end_before_evaluation(
            self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float, batch_nr_examples: int,
            epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, avg_throughput: float):
        if self.last_taken_step_start_time_within_epoch is not None:
            self.cur_epoch_time += (datetime.datetime.now() - self.last_taken_step_start_time_within_epoch)
            self.last_taken_step_start_time_within_epoch = None
        self.eval_start_time = datetime.datetime.now()

    def step_end_after_evaluation(
            self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float, batch_nr_examples: int,
            epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, validation_loss: float,
            validation_metrics_results: Dict[str, float], avg_throughput: float):
        eval_time = (datetime.datetime.now() - self.eval_start_time).total_seconds() / 60
        self.eval_start_time = None
        epoch_results = {
            'train/epoch': epoch_nr,
            'train/batch': step_nr,
            'train/epoch_avg_loss': epoch_avg_loss,
            'train/moving_win_loss': epoch_moving_win_loss.get_window_avg(),
            'train/avg_throughput': avg_throughput,
            'train/epoch_duration': self.cur_epoch_time.total_seconds() / 60,
            'val/loss': validation_loss,
            'val/duration': eval_time,
            **{f'val/metrics/{k}': v
               for k, v in _flatten_structured_object(validation_metrics_results, fields_delimiter='/').items()}
        }
        wandb.log(epoch_results)

    def epoch_end(
            self, epoch_nr: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, avg_throughput: float):
        self.cur_epoch_time = None
        self.last_taken_step_start_time_within_epoch = None
        self.eval_start_time = None
        self.partial_epoch_results_last_msg_time = None

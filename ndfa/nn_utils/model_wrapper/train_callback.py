__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2020-05-16"

from typing import Dict, Tuple

from ndfa.nn_utils.model_wrapper.window_average import WindowAverage


__all__ = ['TrainCallback']


class TrainCallback:
    def epoch_start(
            self, epoch_nr: int, step_nr: int, learning_rates: Tuple[float, ...]):
        pass

    def epoch_end_before_evaluation(
            self, epoch_nr: int, step_nr: int, epoch_avg_loss: float,
            epoch_moving_win_loss: WindowAverage, avg_throughput: float,
            learning_rates: Tuple[float, ...]):
        pass

    def epoch_end_after_evaluation(
            self, epoch_nr: int, step_nr: int, epoch_avg_loss: float,
            epoch_moving_win_loss: WindowAverage, validation_loss: float,
            validation_metrics_results: Dict[str, float], avg_throughput: float,
            learning_rates: Tuple[float, ...]):
        pass

    def epoch_end(
            self, epoch_nr: int, step_nr: int, epoch_avg_loss: float,
            epoch_moving_win_loss: WindowAverage, avg_throughput: float,
            learning_rates: Tuple[float, ...]):
        pass

    def step_start(
            self, epoch_nr: int, step_nr: int, batch_nr: int, nr_batches_in_epoch: int,
            avg_throughput: float, learning_rates: Tuple[float, ...]):
        pass

    def step_end(
            self, epoch_nr: int, step_nr: int, batch_nr: int, nr_batches_in_epoch: int,
            batch_loss: float, batch_nr_examples: int, epoch_avg_loss: float,
            epoch_moving_win_loss: WindowAverage, avg_throughput: float,
            learning_rates: Tuple[float, ...]):
        pass

    def step_end_before_evaluation(
            self, epoch_nr: int, step_nr: int, batch_nr: int, nr_batches_in_epoch: int,
            batch_loss: float, batch_nr_examples: int, epoch_avg_loss: float,
            epoch_moving_win_loss: WindowAverage, avg_throughput: float,
            learning_rates: Tuple[float, ...]):
        pass

    def step_end_after_evaluation(
            self, epoch_nr: int, step_nr: int, batch_nr: int, nr_batches_in_epoch: int, batch_loss: float,
            batch_nr_examples: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage,
            validation_loss: float, validation_metrics_results: Dict[str, float],
            avg_throughput: float, learning_rates: Tuple[float, ...]):
        pass

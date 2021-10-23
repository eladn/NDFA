from typing import Dict

from ndfa.nn_utils.model_wrapper.window_average import WindowAverage


__all__ = ['TrainCallback']


class TrainCallback:
    def epoch_start(self, epoch_nr: int):
        pass

    def epoch_end_before_evaluation(
            self, epoch_nr: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, avg_throughput: float):
        pass

    def epoch_end_after_evaluation(
            self, epoch_nr: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage,
            validation_loss: float, validation_metrics_results: Dict[str, float], avg_throughput: float):
        pass

    def epoch_end(
            self, epoch_nr: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, avg_throughput: float):
        pass

    def step_start(
            self, epoch_nr: int, step_nr: int, nr_steps: int, avg_throughput: float):
        pass

    def step_end(
            self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float, batch_nr_examples: int,
            epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, avg_throughput: float):
        pass

    def step_end_before_evaluation(
            self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float, batch_nr_examples: int,
            epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage, avg_throughput: float):
        pass

    def step_end_after_evaluation(
            self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float,
            batch_nr_examples: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage,
            validation_loss: float, validation_metrics_results: Dict[str, float], avg_throughput: float):
        pass

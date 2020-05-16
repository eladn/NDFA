import time
from notify_run import Notify
from typing import Dict, Optional

from ddfa.nn_utils.train_callback import TrainCallback
from ddfa.nn_utils.window_average import WindowAverage


__all__ = ['NotifyCallback']


class NotifyCallback(TrainCallback):
    def __init__(self, notify_every_mins: Optional[int] = 5):
        self.notify_every_mins = notify_every_mins

        self.low_verbosity_notify = Notify()
        self.low_verbosity_notify.register()
        print(f'Low verbosity notify endpoint: `{self.low_verbosity_notify.endpoint}`.')

        self.high_verbosity_notify = None
        self.high_verbosity_last_msg_time = None
        if notify_every_mins is not None:
            assert notify_every_mins > 0
            self.high_verbosity_notify = Notify()
            self.high_verbosity_notify.register()
            print(f'High verbosity notify endpoint: `{self.high_verbosity_notify.endpoint}`.')

    def step_end(self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float, batch_nr_examples: int,
                 epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage):
        if self.high_verbosity_notify is None:
            return
        if self.high_verbosity_last_msg_time is None or \
                self.high_verbosity_last_msg_time + self.notify_every_mins >= time.time():
            self.low_verbosity_notify.send(
                f'Ep {epoch_nr} step {step_nr}/{nr_steps}: ')
            self.high_verbosity_last_msg_time = time.time()

    def epoch_end_after_evaluation(self, epoch_nr: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage,
                                   validation_loss: float, validation_metrics_results: Dict[str, float]):
        msg = f'Completed performing training & evaluation for epoch #{epoch_nr}.' \
              f'\n\t validation loss: {validation_loss:.4f}' \
              f'\n\t validation metrics: {validation_metrics_results}'
        if self.high_verbosity_notify:
            self.high_verbosity_notify.send(msg)
        self.low_verbosity_notify.send(msg)

    def step_end_after_evaluation(self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float,
                                  batch_nr_examples: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage,
                                  validation_loss: float, validation_metrics_results: Dict[str, float]):
        msg = f'Completed performing evaluation DURING epoch #{epoch_nr} (after step {step_nr}/{nr_steps}).' \
              f'\n\t validation loss: {validation_loss:.4f}' \
              f'\n\t validation metrics: {validation_metrics_results}'
        if self.high_verbosity_notify:
            self.high_verbosity_notify.send(msg)
        self.low_verbosity_notify.send(msg)

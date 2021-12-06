__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2020-05-16"

import time
from typing import Dict, Optional, Tuple

from ndfa.nn_utils.model_wrapper.train_callback import TrainCallback
from ndfa.nn_utils.model_wrapper.window_average import WindowAverage
from ndfa.nn_utils.model_wrapper.async_notify import AsyncNotifyChannel


__all__ = ['NotifyCallback']


class NotifyCallback(TrainCallback):
    def __init__(self, notify_every_mins: Optional[int] = 5, send_only_last_pending_msg: bool = True):
        self.notify_every_mins = notify_every_mins

        self.low_verbosity_notify = AsyncNotifyChannel(send_only_last_pending_msg=send_only_last_pending_msg)
        self.low_verbosity_notify.start()
        print(f'Low verbosity notify endpoint: `{self.low_verbosity_notify.notify_endpoint.link}`.')

        self.high_verbosity_notify = None
        self.high_verbosity_last_msg_time = None
        if notify_every_mins is not None:
            assert notify_every_mins > 0
            self.high_verbosity_notify = AsyncNotifyChannel(send_only_last_pending_msg=send_only_last_pending_msg)
            self.high_verbosity_notify.start()
            print(f'High verbosity notify endpoint: `{self.high_verbosity_notify.notify_endpoint.link}`.')

    def __del__(self):
        self.low_verbosity_notify.close()
        if self.high_verbosity_notify is not None:
            self.high_verbosity_notify.close()

    def step_end(
            self, epoch_nr: int, step_nr: int, batch_nr: int, nr_batches_in_epoch: int,
            batch_loss: float, batch_nr_examples: int, epoch_avg_loss: float,
            epoch_moving_win_loss: WindowAverage, avg_throughput: float,
            learning_rates: Tuple[float, ...]):
        if self.high_verbosity_notify is None:
            return
        if self.high_verbosity_last_msg_time is None or \
                self.high_verbosity_last_msg_time + self.notify_every_mins * 60 <= time.time():
            msg = f'Ep {epoch_nr} - step {step_nr} - batch {batch_nr}/{nr_batches_in_epoch} ' \
                  f'[{int(100 * batch_nr / nr_batches_in_epoch)}%]: '\
                  f'loss (epoch avg): {epoch_avg_loss:.4f} -- '\
                  f'loss (win avg): {epoch_moving_win_loss.get_window_avg():.4f} -- '\
                  f'loss (win stbl avg): {epoch_moving_win_loss.get_window_avg_wo_outliers():.4f}'
            try:
                self.high_verbosity_notify.send(msg)
            except:
                pass
            self.high_verbosity_last_msg_time = time.time()

    def epoch_end_after_evaluation(
            self, epoch_nr: int, step_nr: int, epoch_avg_loss: float,
            epoch_moving_win_loss: WindowAverage, validation_loss: float,
            validation_metrics_results: Dict[str, float], avg_throughput: float,
            learning_rates: Tuple[float, ...]):
        # TODO: For pretty printing the evaluation metric results:
        #       https://stackoverflow.com/questions/44356693/pprint-with-custom-float-formats
        msg = f'Completed performing training & evaluation for epoch #{epoch_nr} (step {step_nr}).' \
              f'\n\t train loss (epoch avg): {epoch_avg_loss:.4f}' \
              f'\n\t loss (win avg): {epoch_moving_win_loss.get_window_avg():.4f}' \
              f'\n\t loss (win stbl avg): {epoch_moving_win_loss.get_window_avg_wo_outliers():.4f}' \
              f'\n\t validation loss: {validation_loss:.4f}' \
              f'\n\t validation metrics: {validation_metrics_results}'
        if self.high_verbosity_notify:
            try:
                self.high_verbosity_notify.send(msg)
            except:
                pass
        try:
            self.low_verbosity_notify.send(msg)
        except:
            pass

    def step_end_after_evaluation(
            self, epoch_nr: int, step_nr: int, batch_nr: int, nr_batches_in_epoch: int, batch_loss: float,
            batch_nr_examples: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage,
            validation_loss: float, validation_metrics_results: Dict[str, float],
            avg_throughput: float, learning_rates: Tuple[float, ...]):
        # TODO: For pretty printing the evaluation metric results:
        #       https://stackoverflow.com/questions/44356693/pprint-with-custom-float-formats
        msg = f'Completed performing evaluation DURING epoch #{epoch_nr} (step {step_nr}) ' \
              f'(batch {batch_nr}/{nr_batches_in_epoch} [{int(100 * batch_nr / nr_batches_in_epoch)}%]).' \
              f'\n\t train loss (epoch avg): {epoch_avg_loss:.4f}' \
              f'\n\t loss (win avg): {epoch_moving_win_loss.get_window_avg():.4f}' \
              f'\n\t loss (win stbl avg): {epoch_moving_win_loss.get_window_avg_wo_outliers():.4f}' \
              f'\n\t validation loss: {validation_loss:.4f}' \
              f'\n\t validation metrics: {validation_metrics_results}'
        if self.high_verbosity_notify:
            try:
                self.high_verbosity_notify.send(msg)
            except:
                pass
        try:
            self.low_verbosity_notify.send(msg)
        except:
            pass

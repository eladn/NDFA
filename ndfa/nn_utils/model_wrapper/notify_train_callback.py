import time
from typing import Dict, Optional

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

    def step_end(self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float, batch_nr_examples: int,
                 epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage):
        if self.high_verbosity_notify is None:
            return
        if self.high_verbosity_last_msg_time is None or \
                self.high_verbosity_last_msg_time + self.notify_every_mins * 60 <= time.time():
            msg = f'Ep {epoch_nr} step {step_nr}/{nr_steps} [{int(100 * step_nr / nr_steps)}%]: '\
                  f'loss (epoch avg): {epoch_avg_loss:.4f} -- '\
                  f'loss (win avg): {epoch_moving_win_loss.get_window_avg():.4f} -- '\
                  f'loss (win stbl avg): {epoch_moving_win_loss.get_window_avg_wo_outliers():.4f}'
            try:
                self.high_verbosity_notify.send(msg)
            except:
                pass
            self.high_verbosity_last_msg_time = time.time()

    def epoch_end_after_evaluation(self, epoch_nr: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage,
                                   validation_loss: float, validation_metrics_results: Dict[str, float]):
        # TODO: For pretty printing the evaluation metric results:
        #       https://stackoverflow.com/questions/44356693/pprint-with-custom-float-formats
        msg = f'Completed performing training & evaluation for epoch #{epoch_nr}.' \
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

    def step_end_after_evaluation(self, epoch_nr: int, step_nr: int, nr_steps: int, batch_loss: float,
                                  batch_nr_examples: int, epoch_avg_loss: float, epoch_moving_win_loss: WindowAverage,
                                  validation_loss: float, validation_metrics_results: Dict[str, float]):
        # TODO: For pretty printing the evaluation metric results:
        #       https://stackoverflow.com/questions/44356693/pprint-with-custom-float-formats
        msg = f'Completed performing evaluation DURING epoch #{epoch_nr} ' \
              f'(after step {step_nr}/{nr_steps} [{int(100 * step_nr / nr_steps)}%]).' \
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

import time
from notify_run import Notify
from typing import Dict, Optional
import multiprocessing as mp

from ddfa.nn_utils.train_callback import TrainCallback
from ddfa.nn_utils.window_average import WindowAverage


__all__ = ['NotifyCallback']


class CloseNotifyProcessAction:
    pass


def _notify_process_worker(msgs_queue: mp.Queue, notify_worker_status_queue: mp.Queue):
    notify = Notify()
    notify.register()
    msgs_queue.put(notify.endpoint, block=True)
    while True:
        msg_to_send = msgs_queue.get(block=True)
        # maybe clear queue and get only last msg? (with nowait method within a loop)
        if isinstance(msg_to_send, CloseNotifyProcessAction):
            break
        assert isinstance(msg_to_send, str)
        try:
            notify.send(msg_to_send)
        except Exception as e:
            print(f'Could not send notify message')


class NotifyChannel:
    def __init__(self):
        # self.pipe_end, worker_pipe_end = mp.Pipe()
        self.msgs_queue = mp.Queue()
        self.notify_worker_status_queue = mp.Queue()
        self.notify_process = mp.Process(
            target=_notify_process_worker, args=(self.msgs_queue, self.notify_worker_status_queue))
        self.notify_endpoint = None

    def start(self):
        self.notify_endpoint = self.msgs_queue.get(block=True)
        self.notify_process.start()

    def send(self, msg):
        self.msgs_queue.put_nowait(msg)

    def close(self):
        self.msgs_queue.put(CloseNotifyProcessAction, block=True)
        self.notify_process.join()
        self.notify_process.close()
        self.msgs_queue.close()
        self.notify_worker_status_queue.close()


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

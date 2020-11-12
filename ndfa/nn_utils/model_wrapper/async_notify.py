import multiprocessing as mp


__all__ = ['AsyncNotifyChannel']


class CloseNotifyProcessAction:
    pass


class NotifyError:
    def __init__(self, err):
        self.err = err


class NotifyEndpoint:
    def __init__(self, link: str):
        self.link = link


def _notify_process_worker(
        msgs_queue: mp.Queue,
        send_only_last_pending_msg: bool = False):
    try:
        from notify_run import Notify
        notify = Notify()
        notify.register()
    except Exception as e:
        msgs_queue.put(NotifyError(e), block=True)
        return
    msgs_queue.put(NotifyEndpoint(notify.endpoint), block=True)
    while True:
        pending_msg = msgs_queue.get(block=True)
        if isinstance(pending_msg, CloseNotifyProcessAction):
            return
        finish_after_current_msg = False
        if send_only_last_pending_msg:
            while True:
                try:
                    newer_pending_msg = msgs_queue.get(block=False)
                    if isinstance(pending_msg, CloseNotifyProcessAction):
                        finish_after_current_msg = True
                        break
                    else:
                        pending_msg = newer_pending_msg
                except Exception:
                    break
        assert isinstance(pending_msg, str)
        try:
            notify.send(pending_msg)
        except Exception as e:
            print(f'Could not send notify message. Got {type(e)} error: {e}')
        if finish_after_current_msg:
            return


class AsyncNotifyChannel:
    def __init__(self, send_only_last_pending_msg: bool = False):
        # self.pipe_end, worker_pipe_end = mp.Pipe()
        self.send_only_last_pending_msg = send_only_last_pending_msg
        self.msgs_queue = mp.Queue()
        self.notify_process = mp.Process(
            target=_notify_process_worker,
            args=(self.msgs_queue, self.send_only_last_pending_msg))
        self.notify_endpoint = None

    @property
    def is_initiated(self) -> bool:
        return self.notify_endpoint is not None

    def start(self):
        assert not self.is_initiated
        self.notify_process.start()
        self.notify_endpoint = self.msgs_queue.get(block=True)

    def send(self, msg):
        assert self.is_initiated
        self.msgs_queue.put_nowait(msg)

    def close(self):
        assert self.is_initiated
        self.msgs_queue.put(CloseNotifyProcessAction(), block=True)
        self.notify_process.join()
        self.notify_process.close()
        self.msgs_queue.close()

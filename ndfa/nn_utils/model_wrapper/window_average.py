import numpy as np
from typing import Optional


__all__ = ['WindowAverage']


class CyclicAppendOnlyBuffer:
    def __init__(self, buffer_size: int):
        assert buffer_size > 0
        self._buffer_size = buffer_size
        self._nr_items = 0
        self._buffer = [None] * buffer_size
        self._next_insert_idx = 0

    def insert(self, item):
        self._buffer[self._next_insert_idx] = item
        self._next_insert_idx = (self._next_insert_idx + 1) % self._buffer_size
        self._nr_items = min(self._nr_items + 1, self._buffer_size)

    def get_all_items(self):
        return (self._buffer[self._next_insert_idx:] if self._nr_items == self._buffer_size else []) + \
               self._buffer[:self._next_insert_idx]

    def __len__(self) -> int:
        return self._nr_items


class WindowAverage:
    def __init__(self, max_window_size: int = 5):
        self._max_window_size = max_window_size
        self._items_buffer = CyclicAppendOnlyBuffer(buffer_size=max_window_size)

    def update(self, number: float):
        self._items_buffer.insert(number)

    def get_window_avg(self, sub_window_size: Optional[int] = None):
        assert sub_window_size is None or sub_window_size <= self._max_window_size
        window_size = self._max_window_size if sub_window_size is None else sub_window_size
        return np.average(self._items_buffer.get_all_items()[:window_size])

    def get_window_avg_wo_outliers(self, nr_outliers: int = 3, sub_window_size: Optional[int] = None):
        assert sub_window_size is None or sub_window_size <= self._max_window_size
        window_size = self._max_window_size if sub_window_size is None else sub_window_size
        assert nr_outliers < window_size
        nr_outliers_wrt_windows_size_max_ratio = nr_outliers / window_size
        assert nr_outliers_wrt_windows_size_max_ratio < 1
        eff_window_size = min(len(self._items_buffer), window_size)
        if eff_window_size < window_size and nr_outliers >= eff_window_size * nr_outliers_wrt_windows_size_max_ratio:
            nr_outliers = int(eff_window_size * nr_outliers_wrt_windows_size_max_ratio)
        assert nr_outliers <= eff_window_size * nr_outliers_wrt_windows_size_max_ratio
        items = np.array(self._items_buffer.get_all_items()[:eff_window_size])
        if nr_outliers > 0:
            median = np.median(items)
            dist_from_median = np.abs(items - median)
            outliers_indices = np.argpartition(dist_from_median, -nr_outliers)[-nr_outliers:]
            items[outliers_indices] = np.nan
        return np.nanmean(items)

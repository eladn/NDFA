import numpy as np
from typing import Dict, Union


__all__ = ['OnlineMeanVarianceAccumulators']


class OnlineMeanVarianceAccumulators:
    def __init__(self):
        self.mean: float = np.nan
        self.variance: float = np.nan
        self.nr_points = 0
        self.nr_nans = 0
        self.nr_pos_infs = 0
        self.nr_neg_infs = 0

    def insert_point(self, x: float):
        if not np.isfinite(x):
            if np.isposinf(x):
                self.nr_pos_infs += 1
            elif np.isneginf(x):
                self.nr_neg_infs += 1
            elif np.nan(x):
                self.nr_nans += 1
            else:
                assert False
            return
        assert np.isfinite(x)
        if self.nr_points == 0:
            self.mean = x
            self.variance = 0
            self.nr_points = 1
        else:
            self.mean, self.variance = self._perform_update_step(
                x=x, last_mean=self.mean, last_variance=self.variance, last_nr_points=self.nr_points)
            self.nr_points += 1

    @property
    def std(self):
        return np.sqrt(self.variance)

    @classmethod
    def _perform_update_step(cls, x, last_mean, last_variance, last_nr_points):
        dx = x - last_mean
        next_mean = last_mean + dx / (last_nr_points + 1)
        # This is population-estimator (divide by `n`):
        # next_variance = (last_nr_points / (last_nr_points + 1)) * (last_variance + dx ** 2 / (last_nr_points + 1))
        # This is an unbiased sample-estimator (divide by `n-1`):
        next_variance = \
            last_variance + ((last_nr_points / (last_nr_points + 1)) * dx ** 2 - last_variance) / last_nr_points
        return next_mean, next_variance

    def __len__(self):
        return self.nr_points

    def generate_printable_stats_dict(self) -> Dict[str, Union[int, float]]:
        stats_dict = {}
        if self.nr_points:
            stats_dict['mean'] = round(self.mean, 2)
            stats_dict['std'] = round(self.std, 2)
        if self.nr_nans:
            stats_dict['#NaNs'] = self.nr_nans
        if self.nr_pos_infs:
            stats_dict['#inf'] = self.nr_pos_infs
        if self.nr_neg_infs:
            stats_dict['#-inf'] = self.nr_neg_infs
        return stats_dict

import abc
from typing import Dict


__all__ = ['EvaluationMetric']


class EvaluationMetric(abc.ABC):
    @abc.abstractmethod
    def update(self, y_hat, target):
        ...

    @abc.abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        ...

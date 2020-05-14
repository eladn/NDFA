from collections import defaultdict, OrderedDict
from typing import Dict, Set

from ddfa.code_tasks.code_task_base import EvaluationMetric


__all__ = ['SymbolsSetEvaluationMetric']


class SymbolsSetEvaluationMetric(EvaluationMetric):
    def __init__(self):
        self.tot_nr_true_positives = 0
        self.tot_nr_false_positives = 0
        self.tot_nr_false_negatives = 0
        self.nr_true_positives_distr = defaultdict(int)
        self.nr_false_positives_distr = defaultdict(int)
        self.nr_false_negatives_distr = defaultdict(int)
        self.nr_predictions = 0
        self.tot_nr_predicted_symbols = 0
        self.tot_nr_groundtrue_target_symbols = 0
        self.tot_nr_perfectly_correct_predictions = 0
        self.tot_nr_perfectly_correct_predictions_per_nr_groundtrue_target_symbols = defaultdict(int)
        self.nr_examples_per_nr_groundtrue_target_symbols = defaultdict(int)

    def update(self, example_pred_symbols_indices: Set[int], example_target_symbols_indices: Set[int]):
        nr_tps = len(example_target_symbols_indices & example_pred_symbols_indices)
        nr_fps = len(example_pred_symbols_indices - example_target_symbols_indices)
        nr_fns = len(example_target_symbols_indices - example_pred_symbols_indices)
        self.tot_nr_true_positives += nr_tps
        self.tot_nr_false_positives += nr_fps
        self.tot_nr_false_negatives += nr_fns
        self.nr_true_positives_distr[nr_tps] += 1
        self.nr_false_positives_distr[nr_fps] += 1
        self.nr_false_negatives_distr[nr_fns] += 1
        self.tot_nr_predicted_symbols += len(example_pred_symbols_indices)
        self.tot_nr_groundtrue_target_symbols += len(example_target_symbols_indices)
        if example_pred_symbols_indices == example_target_symbols_indices:
            self.tot_nr_perfectly_correct_predictions += 1
            self.tot_nr_perfectly_correct_predictions_per_nr_groundtrue_target_symbols[len(example_target_symbols_indices)] += 1
        self.nr_examples_per_nr_groundtrue_target_symbols[len(example_target_symbols_indices)] += 1
        self.nr_predictions += 1

    def get_metrics(self) -> Dict[str, float]:
        return {
            'acc': self.accuracy,
            'f1': self.f1,
            'precision': self.precision,
            'recall': self.recall,
            'acc per #gtSymbols': self.accuracy_per_nr_groundtrue_target_symbols,
            'distr(tp)': self.true_positive_distr,
            'distr(fp)': self.false_positive_distr,
            'distr(fn)': self.false_negative_distr,
            'tp/all': self.true_positive_total_rate,
            'fp/all': self.true_positive_total_rate,
            'fn/all': self.true_positive_total_rate,
            '#tp/pred': self.true_positives_per_prediction_avg,
            '#fp/pred': self.false_positives_per_prediction_avg,
            '#fn/pred': self.false_negatives_per_prediction_avg,
            'avg(#predSymbols)': self.tot_nr_predicted_symbols / self.nr_predictions,
            'avg(#gtSymbols)': self.tot_nr_groundtrue_target_symbols / self.nr_predictions
        }

    @property
    def accuracy(self) -> float:
        return self.tot_nr_perfectly_correct_predictions / self.nr_predictions

    @property
    def accuracy_per_nr_groundtrue_target_symbols(self) -> Dict[int, float]:
        distr = [
            (nr_gt_symbols, nr_correct / self.nr_examples_per_nr_groundtrue_target_symbols[nr_gt_symbols])
            for nr_gt_symbols, nr_correct in self.tot_nr_perfectly_correct_predictions_per_nr_groundtrue_target_symbols.items()]
        distr.sort(key=lambda x: x[0])
        return OrderedDict(distr)

    @property
    def true_positive_distr(self) -> Dict[int, float]:
        all_count = sum(self.nr_true_positives_distr.values())
        distr = [(num, 100 * total_count / all_count)
                 for num, total_count in self.nr_true_positives_distr.items()]
        distr.sort(key=lambda x: x[0])
        return OrderedDict(distr)

    @property
    def false_positive_distr(self) -> Dict[int, float]:
        all_count = sum(self.nr_true_positives_distr.values())
        distr = [(num, 100 * total_count / all_count)
                 for num, total_count in self.nr_false_positives_distr.items()]
        distr.sort(key=lambda x: x[0])
        return OrderedDict(distr)

    @property
    def false_negative_distr(self) -> Dict[int, float]:
        all_count = sum(self.nr_true_positives_distr.values())
        distr = [(num, 100 * total_count / all_count)
                 for num, total_count in self.nr_false_negatives_distr.items()]
        distr.sort(key=lambda x: x[0])
        return OrderedDict(distr)

    @property
    def total_nr_tokens(self) -> int:
        return self.tot_nr_true_positives + self.tot_nr_false_positives + self.tot_nr_false_negatives

    @property
    def true_positive_total_rate(self) -> float:
        return self.tot_nr_true_positives / (self.total_nr_tokens + np.finfo(np.float32).eps)

    @property
    def false_positive_total_rate(self) -> float:
        return self.tot_nr_false_positives / (self.total_nr_tokens + np.finfo(np.float32).eps)

    @property
    def false_negative_total_rate(self) -> float:
        return self.tot_nr_false_negatives / (self.total_nr_tokens + np.finfo(np.float32).eps)

    @property
    def true_positives_per_prediction_avg(self) -> float:
        return self.tot_nr_true_positives / (self.nr_predictions + np.finfo(np.float32).eps)

    @property
    def false_positives_per_prediction_avg(self) -> float:
        return self.tot_nr_false_positives / (self.nr_predictions + np.finfo(np.float32).eps)

    @property
    def false_negatives_per_prediction_avg(self) -> float:
        return self.tot_nr_false_negatives / (self.nr_predictions + np.finfo(np.float32).eps)

    @property
    def precision(self) -> float:
        return self.tot_nr_true_positives / (self.tot_nr_true_positives + self.tot_nr_false_positives + np.finfo(np.float32).eps)

    @property
    def recall(self) -> float:
        return self.tot_nr_true_positives / (self.tot_nr_true_positives + self.tot_nr_false_negatives + np.finfo(np.float32).eps)

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        return (2 * precision * recall) / (precision + recall + np.finfo(np.float32).eps)

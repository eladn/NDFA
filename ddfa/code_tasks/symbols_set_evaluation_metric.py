from collections import defaultdict, OrderedDict
from typing import Dict, Set, List
import numpy as np

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
        self.tot_nr_perfectly_correct_unordered_predictions = 0
        self.tot_nr_perfectly_correct_ordered_predictions = 0
        self.tot_nr_perfectly_correct_unordered_predictions_per_nr_groundtrue_target_symbols = defaultdict(int)
        self.tot_nr_perfectly_correct_ordered_predictions_per_nr_groundtrue_target_symbols = defaultdict(int)
        self.nr_examples_per_nr_groundtrue_target_symbols = defaultdict(int)

    def update(self, example_pred_symbols_indices: List[int], example_target_symbols_indices: List[int]):
        example_pred_symbols_indices_set: Set[int] = set(example_pred_symbols_indices)
        example_target_symbols_indices_set: Set[int] = set(example_target_symbols_indices)
        nr_tps = len(example_target_symbols_indices_set & example_pred_symbols_indices_set)
        nr_fps = len(example_pred_symbols_indices_set - example_target_symbols_indices_set)
        nr_fns = len(example_target_symbols_indices_set - example_pred_symbols_indices_set)
        self.tot_nr_true_positives += nr_tps
        self.tot_nr_false_positives += nr_fps
        self.tot_nr_false_negatives += nr_fns
        self.nr_true_positives_distr[nr_tps] += 1
        self.nr_false_positives_distr[nr_fps] += 1
        self.nr_false_negatives_distr[nr_fns] += 1
        self.tot_nr_predicted_symbols += len(example_pred_symbols_indices_set)
        self.tot_nr_groundtrue_target_symbols += len(example_target_symbols_indices_set)
        if example_pred_symbols_indices == example_target_symbols_indices:
            self.tot_nr_perfectly_correct_ordered_predictions += 1
            self.tot_nr_perfectly_correct_ordered_predictions_per_nr_groundtrue_target_symbols[
                len(example_target_symbols_indices_set)] += 1
        if example_pred_symbols_indices_set == example_target_symbols_indices_set:
            self.tot_nr_perfectly_correct_unordered_predictions += 1
            self.tot_nr_perfectly_correct_unordered_predictions_per_nr_groundtrue_target_symbols[
                len(example_target_symbols_indices_set)] += 1
        self.nr_examples_per_nr_groundtrue_target_symbols[len(example_target_symbols_indices_set)] += 1
        self.nr_predictions += 1

    def get_metrics(self) -> Dict[str, float]:
        return {
            'acc (unordered)': self.accuracy_unordered,
            'acc (ordered)': self.accuracy_ordered,
            'f1': self.f1,
            'precision': self.precision,
            'recall': self.recall,
            'acc (unordered) per #gtSymbols': self.accuracy_unordered_per_nr_groundtrue_target_symbols,
            'acc (ordered) per #gtSymbols': self.accuracy_ordered_per_nr_groundtrue_target_symbols,
            'distr(#tp)': self.true_positive_distr,
            'distr(#fp)': self.false_positive_distr,
            'distr(#fn)': self.false_negative_distr,
            '%tp/(tp+fp+fn)': self.true_positive_total_rate,
            '%fp/(tp+fp+fn)': self.false_positive_total_rate,
            '%fn/(tp+fp+fn)': self.false_negative_total_rate,
            'avg_pred(#tp)': self.true_positives_per_prediction_avg,
            'avg_pred(#fp)': self.false_positives_per_prediction_avg,
            'avg_pred(#fn)': self.false_negatives_per_prediction_avg,
            'avg(#predSymbols)': self.tot_nr_predicted_symbols / self.nr_predictions,
            'avg(#gtSymbols)': self.tot_nr_groundtrue_target_symbols / self.nr_predictions
        }

    @property
    def accuracy_unordered(self) -> float:
        return self.tot_nr_perfectly_correct_unordered_predictions / self.nr_predictions

    @property
    def accuracy_ordered(self) -> float:
        return self.tot_nr_perfectly_correct_ordered_predictions / self.nr_predictions

    @property
    def accuracy_unordered_per_nr_groundtrue_target_symbols(self) -> Dict[int, float]:
        distr = [
            (nr_gt_symbols, nr_correct / self.nr_examples_per_nr_groundtrue_target_symbols[nr_gt_symbols])
            for nr_gt_symbols, nr_correct in self.tot_nr_perfectly_correct_unordered_predictions_per_nr_groundtrue_target_symbols.items()]
        distr.sort(key=lambda x: x[0])
        return OrderedDict(distr)

    @property
    def accuracy_ordered_per_nr_groundtrue_target_symbols(self) -> Dict[int, float]:
        distr = [
            (nr_gt_symbols, nr_correct / self.nr_examples_per_nr_groundtrue_target_symbols[nr_gt_symbols])
            for nr_gt_symbols, nr_correct in self.tot_nr_perfectly_correct_ordered_predictions_per_nr_groundtrue_target_symbols.items()]
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

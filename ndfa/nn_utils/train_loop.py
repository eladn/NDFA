import sys
import time
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Callable, List, Type, Tuple, Dict, Collection

from ndfa.nn_utils.window_average import WindowAverage
from ndfa.nn_utils.train_callback import TrainCallback
from ndfa.code_tasks.evaluation_metric_base import EvaluationMetric


__all__ = ['fit', 'evaluate']


def perform_loss_step_for_batch(device, x_batch: torch.Tensor, y_batch: torch.Tensor, model: nn.Module,
                                criterion: nn.Module, optimizer: Optional[Optimizer] = None,
                                batch_idx: Optional[int] = None, nr_batches: Optional[int] = None,
                                nr_gradient_accumulation_steps: int = 1, dbg_test_grads: bool = False,
                                lazy_move_to_device_history=None):
    if lazy_move_to_device_history is None:
        lazy_move_to_device_history = {'x': {}, 'y': {}}
    else:
        if 'x' not in lazy_move_to_device_history:
            lazy_move_to_device_history['x'] = {}
        if 'y' not in lazy_move_to_device_history:
            lazy_move_to_device_history['y'] = {}

    # torch.cuda.empty_cache()  # this avoids OOM on bigger bsz, but makes all work slowly

    # x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    x_batch = x_batch.lazy_to(device, lazy_move_to_device_history['x']) \
        if hasattr(x_batch, 'lazy_to') else x_batch.to(device)
    y_batch = y_batch.lazy_to(device, lazy_move_to_device_history['y']) \
        if hasattr(y_batch, 'lazy_to') else y_batch.to(device)

    y_pred = model(x_batch, y_batch)
    loss = criterion(y_pred, y_batch) / nr_gradient_accumulation_steps
    if optimizer is not None:
        if dbg_test_grads:
            model.dbg_retain_grads()
        loss.backward()
        if dbg_test_grads:
            model.dbg_test_grads()
        if nr_gradient_accumulation_steps is None or (batch_idx % nr_gradient_accumulation_steps) == nr_gradient_accumulation_steps - 1 or \
                (nr_batches is not None and batch_idx == nr_batches - 1):
            optimizer.step()
            optimizer.zero_grad()
    return y_pred, loss.item() * nr_gradient_accumulation_steps, x_batch.batch_size


class AuxTaskSchedulerDuringTrainEpoch:
    def __init__(self, min_train_epoch_minutes_to_perform_task: float,
                 task_time_consumption_ratio: float, task_duration_est: float):
        self.min_train_epoch_minutes_to_perform_task = min_train_epoch_minutes_to_perform_task
        self.task_time_consumption_ratio = task_time_consumption_ratio
        self.step_nr_of_last_call = None
        self.nr_calls_performed_during_epoch = 0
        self.task_avg_duration = task_duration_est

    def whether_to_call_this_step(
            self, cur_step_nr: int, total_nr_steps_in_epoch: int, train_step_avg_time: float) -> bool:
        if cur_step_nr == total_nr_steps_in_epoch:
            return False
        assert self.task_avg_duration is not None

        nr_calls_during_train_epoch = self.calc_nr_calls_during_train_epoch(
            total_nr_steps_in_epoch=total_nr_steps_in_epoch, train_step_avg_time=train_step_avg_time)
        nr_steps_performed_since_last_call = self.get_nr_steps_performed_since_last_call(cur_step_nr)
        nr_train_steps_to_spread_calls_over = \
            total_nr_steps_in_epoch - cur_step_nr + nr_steps_performed_since_last_call
        nr_remaining_calls_to_perform = \
            nr_calls_during_train_epoch - self.nr_calls_performed_during_epoch
        perform_call_every_nr_train_steps = \
            nr_train_steps_to_spread_calls_over // nr_remaining_calls_to_perform
        do_call_this_step = \
            nr_steps_performed_since_last_call >= perform_call_every_nr_train_steps
        return do_call_this_step

    def calc_nr_calls_during_train_epoch(self, total_nr_steps_in_epoch: int, train_step_avg_time: float):
        train_epoch_avg_time = total_nr_steps_in_epoch * train_step_avg_time
        if self.min_train_epoch_minutes_to_perform_task > train_epoch_avg_time:
            return 0
        total_allowed_task_time_during_epoch = \
            train_epoch_avg_time * self.task_time_consumption_ratio
        return round(total_allowed_task_time_during_epoch / self.task_avg_duration)

    def get_nr_steps_performed_since_last_call(self, cur_step_nr: int) -> int:
        return cur_step_nr if self.step_nr_of_last_call is None else cur_step_nr - self.step_nr_of_last_call

    def report_task_performed(self, cur_step_nr: int, duration: float):
        self.nr_calls_performed_during_epoch += 1
        self.step_nr_of_last_call += cur_step_nr
        self.task_avg_duration = duration if self.task_avg_duration is None else \
            0.8 * self.task_avg_duration + 0.2 * duration


def fit(nr_epochs: int, model: nn.Module, device: torch.device, train_loader: DataLoader,
        valid_loader: Optional[DataLoader], optimizer: Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], criterion: nn.Module = F.nll_loss,
        nr_gradient_accumulation_steps: int = 1,
        save_checkpoint_fn: Optional[Callable[[nn.Module, Optimizer, int, Optional[int]], None]] = None,
        evaluation_metrics_types: Optional[List[Type[EvaluationMetric]]] = None,
        callbacks: Optional[Collection[TrainCallback]] = None,
        evaluation_time_consumption_ratio: float = 1/8,
        min_train_epoch_minutes_to_perform_evaluation_during: float = 10,
        perform_evaluation_before_starting_training: bool = True):
    if callbacks is None:
        callbacks = ()
    model.to(device)

    evaluation_avg_duration = None
    if valid_loader is not None and perform_evaluation_before_starting_training:
        print(f'Performing evaluation (over validation) before starting the training:')
        evaluate_start_time = time.time()
        val_loss, val_metrics_results = evaluate(
            model=model, device=device, valid_loader=valid_loader, criterion=criterion,
            evaluation_metrics_types=evaluation_metrics_types)
        evaluation_avg_duration = time.time() - evaluate_start_time
        print(f'Completed performing evaluation (over validation) before starting the training.'
              f'\n\t validation loss: {val_loss:.4f}'
              f'\n\t validation metrics: {val_metrics_results}')

    train_step_avg_time = None
    avg_throughput = None
    train_lazy_move_to_device_history = {}
    for epoch_nr in range(1, nr_epochs + 1):  # TODO: allow resuming from given epoch number
        print(f'Starting training epoch #{epoch_nr} ..')
        for callback in callbacks:
            callback.epoch_start(epoch_nr=epoch_nr)
        model.train()
        train_epoch_loss_sum = 0
        train_epoch_nr_examples = 0
        train_epoch_avg_loss = 0.0
        evaluation_scheduler = None if valid_loader is None else AuxTaskSchedulerDuringTrainEpoch(
            min_train_epoch_minutes_to_perform_task=min_train_epoch_minutes_to_perform_evaluation_during,
            task_time_consumption_ratio=evaluation_time_consumption_ratio,
            task_duration_est=evaluation_avg_duration)
        train_epoch_window_loss = WindowAverage(max_window_size=50)
        train_data_loader_with_progress = tqdm(train_loader, dynamic_ncols=True, position=0, leave=True)
        nr_steps = len(train_data_loader_with_progress)
        for batch_idx, (x_batch, y_batch) in enumerate(iter(train_data_loader_with_progress)):
            for callback in callbacks:
                callback.step_start(
                    epoch_nr=epoch_nr, step_nr=batch_idx + 1, nr_steps=nr_steps)
            cur_step_start_time = time.time()
            _, batch_loss, batch_nr_examples = perform_loss_step_for_batch(
                device=device, x_batch=x_batch, y_batch=y_batch, model=model,
                criterion=criterion, optimizer=optimizer, batch_idx=batch_idx,
                nr_batches=nr_steps, nr_gradient_accumulation_steps=nr_gradient_accumulation_steps,
                lazy_move_to_device_history=train_lazy_move_to_device_history)
            cur_step_duration = time.time() - cur_step_start_time
            train_step_avg_time = cur_step_duration if train_step_avg_time is None else \
                train_step_avg_time * 0.8 + cur_step_duration * 0.2
            cur_step_throughput = batch_nr_examples / cur_step_duration
            avg_throughput = cur_step_throughput if avg_throughput is None else \
                avg_throughput * 0.8 + cur_step_throughput * 0.2
            train_epoch_loss_sum += batch_loss * batch_nr_examples
            train_epoch_nr_examples += batch_nr_examples
            train_epoch_window_loss.update(batch_loss)
            train_epoch_avg_loss = train_epoch_loss_sum/train_epoch_nr_examples
            train_data_loader_with_progress.set_postfix(
                {'throughput (#ex/s)': f'{avg_throughput:.2f}',
                 'ep': f'{epoch_nr}',
                 'loss (ep avg)': f'{train_epoch_avg_loss:.4f}',
                 'loss (win avg)': f'{train_epoch_window_loss.get_window_avg():.4f}',
                 'loss (win stbl avg)': f'{train_epoch_window_loss.get_window_avg_wo_outliers():.4f}'})

            if valid_loader is not None and evaluation_scheduler.whether_to_call_this_step(
                    cur_step_nr=batch_idx + 1,
                    total_nr_steps_in_epoch=nr_steps,
                    train_step_avg_time=train_step_avg_time):
                for callback in callbacks:
                    callback.step_end_before_evaluation(
                        epoch_nr=epoch_nr, step_nr=batch_idx + 1, nr_steps=nr_steps,
                        batch_loss=batch_loss, batch_nr_examples=batch_nr_examples,
                        epoch_avg_loss=train_epoch_avg_loss, epoch_moving_win_loss=train_epoch_window_loss)

                print(f'Performing evaluation (over validation) DURING epoch #{epoch_nr} '
                      f'(after step {batch_idx + 1}/{nr_steps}):')
                print(file=sys.stderr, flush=True)
                evaluate_start_time = time.time()
                val_loss, val_metrics_results = evaluate(
                    model=model, device=device, valid_loader=valid_loader, criterion=criterion,
                    evaluation_metrics_types=evaluation_metrics_types)
                last_evaluation_duration = time.time() - evaluate_start_time
                evaluation_scheduler.report_task_performed(
                    cur_step_nr=batch_idx + 1, duration=last_evaluation_duration)
                print(f'Completed performing evaluation DURING epoch #{epoch_nr} '
                      f'(after step {batch_idx + 1}/{nr_steps}).'
                      f'\n\t validation loss: {val_loss:.4f}'
                      f'\n\t validation metrics: {val_metrics_results}')

                for callback in callbacks:
                    callback.step_end_after_evaluation(
                        epoch_nr=epoch_nr, step_nr=batch_idx + 1, nr_steps=nr_steps,
                        batch_loss=batch_loss, batch_nr_examples=batch_nr_examples,
                        epoch_avg_loss=train_epoch_avg_loss, epoch_moving_win_loss=train_epoch_window_loss,
                        validation_loss=val_loss, validation_metrics_results=val_metrics_results)

                model.train()

            for callback in callbacks:
                callback.step_end(
                    epoch_nr=epoch_nr, step_nr=batch_idx + 1, nr_steps=nr_steps,
                    batch_loss=batch_loss, batch_nr_examples=batch_nr_examples, epoch_avg_loss=train_epoch_avg_loss,
                    epoch_moving_win_loss=train_epoch_window_loss)

        # TODO: make it a callback of `epoch_end`
        if save_checkpoint_fn is not None:
            save_checkpoint_fn(model, optimizer, epoch_nr, None)

        if valid_loader is not None:
            evaluation_avg_duration = evaluation_scheduler.task_avg_duration
            for callback in callbacks:
                callback.epoch_end_before_evaluation(
                    epoch_nr=epoch_nr, epoch_avg_loss=train_epoch_avg_loss,
                    epoch_moving_win_loss=train_epoch_window_loss)

            print(f'Performing evaluation (over validation) after epoch #{epoch_nr}:')
            val_loss, val_metrics_results = evaluate(
                model=model, device=device, valid_loader=valid_loader, criterion=criterion,
                evaluation_metrics_types=evaluation_metrics_types)
            evaluate_start_time = time.time()
            last_evaluation_duration = time.time() - evaluate_start_time
            evaluation_avg_duration = last_evaluation_duration if evaluation_avg_duration is None else \
                (evaluation_avg_duration * 0.8 + last_evaluation_duration * 0.2)
            print(f'Completed performing training & evaluation for epoch #{epoch_nr}.'
                  f'\n\t validation loss: {val_loss:.4f}'
                  f'\n\t validation metrics: {val_metrics_results}')

            for callback in callbacks:
                callback.epoch_end_after_evaluation(
                    epoch_nr=epoch_nr, epoch_avg_loss=train_epoch_avg_loss,
                    epoch_moving_win_loss=train_epoch_window_loss,
                    validation_loss=val_loss, validation_metrics_results=val_metrics_results)

        for callback in callbacks:
            callback.epoch_end(
                epoch_nr=epoch_nr, epoch_avg_loss=train_epoch_avg_loss, epoch_moving_win_loss=train_epoch_window_loss)

        if lr_scheduler is not None:
            lr_scheduler.step()


def evaluate(model: nn.Module, device: torch.device, valid_loader: DataLoader, criterion: nn.Module,
             evaluation_metrics_types: List[Type[EvaluationMetric]], lazy_move_to_device_history=None) \
        -> Tuple[float, Dict[str, float]]:
    if lazy_move_to_device_history is None:
        lazy_move_to_device_history = {'x': {}, 'y': {}}
    else:
        if 'x' not in lazy_move_to_device_history:
            lazy_move_to_device_history['x'] = {}
        if 'y' not in lazy_move_to_device_history:
            lazy_move_to_device_history['y'] = {}
    model.to(device)
    model.eval()
    metrics = [metric() for metric in evaluation_metrics_types]
    eval_epoch_loss_sum = eval_epoch_nr_examples = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(valid_loader, dynamic_ncols=True, position=0, leave=True):
            x_batch = x_batch.lazy_to(device, lazy_move_to_device_history['x']) \
                if hasattr(x_batch, 'lazy_to') else x_batch.to(device)
            y_batch = y_batch.lazy_to(device, lazy_move_to_device_history['y']) \
                if hasattr(y_batch, 'lazy_to') else y_batch.to(device)
            y_hat = model(x_batch)
            batch_loss = criterion(y_hat, y_batch).item()
            batch_nr_examples = x_batch.batch_size
            for metric in metrics:
                metric.update(y_hat=y_hat.cpu(), target=y_batch.cpu())
            eval_epoch_loss_sum += batch_loss * batch_nr_examples
            eval_epoch_nr_examples += batch_nr_examples
    val_loss = eval_epoch_loss_sum / eval_epoch_nr_examples
    metrics_results = {}
    for metric in metrics:
        metrics_results.update(metric.get_metrics())
    return val_loss, metrics_results


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            data, target = x_batch.to(device), y_batch.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

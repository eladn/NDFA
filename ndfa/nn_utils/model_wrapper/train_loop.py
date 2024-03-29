__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2020-05-07"

import sys
import time
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Callable, List, Type, Tuple, Dict, Collection

from ndfa.nn_utils.model_wrapper.window_average import WindowAverage
from ndfa.nn_utils.model_wrapper.train_callback import TrainCallback
from ndfa.nn_utils.model_wrapper.gradual_lr_warmup_scheduler import GradualLRWarmupScheduler
from ndfa.code_tasks.evaluation_metric_base import EvaluationMetric


__all__ = ['fit', 'evaluate', 'TrainProgressInfo']


def perform_loss_step_for_batch(device, x_batch: torch.Tensor, y_batch: torch.Tensor, model: nn.Module,
                                criterion: nn.Module, optimizer: Optional[Optimizer] = None,
                                batch_idx: Optional[int] = None, nr_batches: Optional[int] = None,
                                nr_gradient_accumulation_steps: int = 1, dbg_test_grads: bool = False,
                                lazy_move_to_device_history=None, gradient_clip_param: Optional[float] = None):
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
    # Note: The criterion loss is assumed to already be scaled by the mini-batch size. That is, different mini-batch
    #       sizes would produce the same loss. Indeed `torch.nn.NLLLoss` is by default set to `reduction='mean'`.
    loss = criterion(y_pred, y_batch) / nr_gradient_accumulation_steps
    optimizer_step_performed = False
    if optimizer is not None:
        if dbg_test_grads:
            model.dbg_retain_grads()
        loss.backward()
        if dbg_test_grads:
            model.dbg_test_grads()
        if nr_gradient_accumulation_steps is None or \
                (batch_idx % nr_gradient_accumulation_steps) == nr_gradient_accumulation_steps - 1 or \
                (nr_batches is not None and batch_idx == nr_batches - 1):
            if gradient_clip_param is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_param)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step_performed = True
    return y_pred, loss.item() * nr_gradient_accumulation_steps, x_batch.batch_size, optimizer_step_performed


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
        if nr_calls_during_train_epoch < 1:
            return False
        nr_steps_performed_since_last_call = self.get_nr_steps_performed_since_last_call(cur_step_nr)
        nr_train_steps_to_spread_calls_over = \
            total_nr_steps_in_epoch - cur_step_nr + nr_steps_performed_since_last_call
        nr_remaining_calls_to_perform = \
            nr_calls_during_train_epoch - self.nr_calls_performed_during_epoch
        if nr_remaining_calls_to_perform < 1:
            return False
        perform_call_every_nr_train_steps = \
            nr_train_steps_to_spread_calls_over // (nr_remaining_calls_to_perform + 1)
        do_call_this_step = \
            nr_steps_performed_since_last_call >= perform_call_every_nr_train_steps
        return do_call_this_step

    def calc_nr_calls_during_train_epoch(self, total_nr_steps_in_epoch: int, train_step_avg_time: float):
        train_epoch_avg_time = total_nr_steps_in_epoch * train_step_avg_time
        if self.min_train_epoch_minutes_to_perform_task * 60 > train_epoch_avg_time:
            return 0
        total_allowed_task_time_during_epoch = \
            train_epoch_avg_time * self.task_time_consumption_ratio
        return round(total_allowed_task_time_during_epoch / self.task_avg_duration)

    def get_nr_steps_performed_since_last_call(self, cur_step_nr: int) -> int:
        return cur_step_nr if self.step_nr_of_last_call is None else cur_step_nr - self.step_nr_of_last_call

    def report_task_performed(self, cur_step_nr: int, duration: float):
        self.nr_calls_performed_during_epoch += 1
        self.step_nr_of_last_call = cur_step_nr
        self.task_avg_duration = duration if self.task_avg_duration is None else \
            0.8 * self.task_avg_duration + 0.2 * duration


@dataclass
class TrainProgressInfo:
    epoch_nr: int = 0


def fit(nr_epochs: int, model: nn.Module, device: torch.device, train_loader: DataLoader,
        valid_loader: Optional[DataLoader], optimizer: Optimizer,
        lr_schedulers: Tuple[torch.optim.lr_scheduler._LRScheduler, ...] = (),
        lr_warmup_scheduler: Optional[GradualLRWarmupScheduler] = None,
        criterion: nn.Module = F.nll_loss,
        nr_gradient_accumulation_steps: int = 1,
        save_checkpoint_fn: Optional[Callable[[nn.Module, Optimizer, int, Optional[int]], None]] = None,
        evaluation_metrics_types: Optional[List[Type[EvaluationMetric]]] = None,
        callbacks: Optional[Collection[TrainCallback]] = None,
        evaluation_time_consumption_ratio: float = 1/8,
        min_train_epoch_minutes_to_perform_evaluation_during: float = 40,
        perform_evaluation_before_starting_training: bool = True,
        gradient_clip_param: Optional[float] = None,
        progress_bar_min_interval_sec: float = 0.1,
        train_progress_info: Optional[TrainProgressInfo] = None):
    if callbacks is None:
        callbacks = ()
    model.to(device)

    evaluation_avg_duration = None
    if valid_loader is not None and perform_evaluation_before_starting_training:
        print(f'Performing evaluation (over validation) before starting the training:')
        evaluate_start_time = time.time()
        val_loss, val_metrics_results = evaluate(
            model=model, device=device, valid_loader=valid_loader, criterion=criterion,
            evaluation_metrics_types=evaluation_metrics_types,
            progress_bar_min_interval_sec=progress_bar_min_interval_sec)
        evaluation_avg_duration = time.time() - evaluate_start_time
        # TODO: For pretty printing the evaluation metric results:
        #       https://stackoverflow.com/questions/44356693/pprint-with-custom-float-formats
        print(f'Completed performing evaluation (over validation) before starting the training.'
              f'\n\t validation loss: {val_loss:.4f}'
              f'\n\t validation metrics: {val_metrics_results}')

    train_step_avg_time = None
    avg_throughput = None
    step_nr = 0
    learning_rates = tuple(param_group['lr'] for param_group in optimizer.param_groups)
    train_lazy_move_to_device_history = {}
    for epoch_nr in range(1, nr_epochs + 1):  # TODO: allow resuming from given epoch number
        print(f'Starting training epoch #{epoch_nr} ..')
        if train_progress_info is not None:
            train_progress_info.epoch_nr = epoch_nr
        for callback in callbacks:
            callback.epoch_start(epoch_nr=epoch_nr, step_nr=step_nr, learning_rates=learning_rates)
        model.train()
        criterion.train()
        train_epoch_nr_examples = 0
        train_epoch_avg_loss = 0.0
        evaluation_scheduler = None if valid_loader is None else AuxTaskSchedulerDuringTrainEpoch(
            min_train_epoch_minutes_to_perform_task=min_train_epoch_minutes_to_perform_evaluation_during,
            task_time_consumption_ratio=evaluation_time_consumption_ratio,
            task_duration_est=evaluation_avg_duration)
        train_epoch_window_loss = WindowAverage(max_window_size=50)
        train_data_loader_with_progress = tqdm(
            train_loader, dynamic_ncols=True, position=0, leave=True, mininterval=progress_bar_min_interval_sec)
        nr_batches_in_epoch = len(train_data_loader_with_progress)
        for batch_idx, (x_batch, y_batch) in enumerate(iter(train_data_loader_with_progress)):
            for callback in callbacks:
                callback.step_start(
                    epoch_nr=epoch_nr, step_nr=step_nr, batch_nr=batch_idx + 1,
                    nr_batches_in_epoch=nr_batches_in_epoch, avg_throughput=avg_throughput,
                    learning_rates=learning_rates)
            cur_step_start_time = time.time()
            # Note: `batch_loss` is the mean reduction over the examples in the batch.
            _, batch_loss, batch_nr_examples, optimizer_step_performed = perform_loss_step_for_batch(
                device=device, x_batch=x_batch, y_batch=y_batch, model=model,
                criterion=criterion, optimizer=optimizer, batch_idx=batch_idx,
                nr_batches=nr_batches_in_epoch, nr_gradient_accumulation_steps=nr_gradient_accumulation_steps,
                lazy_move_to_device_history=train_lazy_move_to_device_history,
                gradient_clip_param=gradient_clip_param)
            if optimizer_step_performed and lr_warmup_scheduler:
                lr_warmup_scheduler.step()
            learning_rates = tuple(param_group['lr'] for param_group in optimizer.param_groups)

            cur_step_duration = time.time() - cur_step_start_time
            train_step_avg_time = cur_step_duration if train_step_avg_time is None else \
                train_step_avg_time * 0.8 + cur_step_duration * 0.2
            cur_step_throughput = batch_nr_examples / cur_step_duration
            avg_throughput = cur_step_throughput if avg_throughput is None else \
                avg_throughput * 0.8 + cur_step_throughput * 0.2
            train_epoch_nr_examples += batch_nr_examples
            # Online mean accumulation update rule:
            #   mu_(n+1) := mu_n + (X_(n+1) - mu_n) / (n+1)
            #   mu_(n+k) := mu_n + (X_(n+1) + ... + X_(n+k) - k * mu_n) / (n+k)
            #   mu_(n+k) := mu_n + (mean(X_(n+1), ..., X_(n+k)) - mu_n) / ((n+k) / k)
            # It should also work if `batch_nr_examples` variates from batch to batch.
            train_epoch_avg_loss += (batch_loss - train_epoch_avg_loss) / (train_epoch_nr_examples / batch_nr_examples)
            train_epoch_window_loss.update(batch_loss)
            train_data_loader_with_progress.set_postfix(
                {'throughput (#ex/s)': f'{avg_throughput:.2f}',
                 'ep': f'{epoch_nr}',
                 'loss (ep avg)': f'{train_epoch_avg_loss:.4f}',
                 'loss (win avg)': f'{train_epoch_window_loss.get_window_avg():.4f}',
                 'loss (win stbl avg)': f'{train_epoch_window_loss.get_window_avg_wo_outliers():.4f}'},
                refresh=False)

            if valid_loader is not None and evaluation_scheduler.whether_to_call_this_step(
                    cur_step_nr=batch_idx + 1,
                    total_nr_steps_in_epoch=nr_batches_in_epoch,
                    train_step_avg_time=train_step_avg_time):
                for callback in callbacks:
                    callback.step_end_before_evaluation(
                        epoch_nr=epoch_nr, step_nr=step_nr, batch_nr=batch_idx + 1,
                        nr_batches_in_epoch=nr_batches_in_epoch,
                        batch_loss=batch_loss, batch_nr_examples=batch_nr_examples,
                        epoch_avg_loss=train_epoch_avg_loss, epoch_moving_win_loss=train_epoch_window_loss,
                        avg_throughput=avg_throughput, learning_rates=learning_rates)

                print(f'Performing evaluation (over validation) DURING epoch #{epoch_nr} '
                      f'(after step {batch_idx + 1}/{nr_batches_in_epoch}):')
                print(file=sys.stderr, flush=True)
                evaluate_start_time = time.time()
                val_loss, val_metrics_results = evaluate(
                    model=model, device=device, valid_loader=valid_loader, criterion=criterion,
                    evaluation_metrics_types=evaluation_metrics_types,
                    progress_bar_min_interval_sec=progress_bar_min_interval_sec)
                last_evaluation_duration = time.time() - evaluate_start_time
                evaluation_scheduler.report_task_performed(
                    cur_step_nr=batch_idx + 1, duration=last_evaluation_duration)
                # TODO: For pretty printing the evaluation metric results:
                #       https://stackoverflow.com/questions/44356693/pprint-with-custom-float-formats
                print(f'Completed performing evaluation DURING epoch #{epoch_nr} '
                      f'(after step {batch_idx + 1}/{nr_batches_in_epoch}).'
                      f'\n\t validation loss: {val_loss:.4f}'
                      f'\n\t validation metrics: {val_metrics_results}')

                for callback in callbacks:
                    callback.step_end_after_evaluation(
                        epoch_nr=epoch_nr, step_nr=step_nr, batch_nr=batch_idx + 1,
                        nr_batches_in_epoch=nr_batches_in_epoch,
                        batch_loss=batch_loss, batch_nr_examples=batch_nr_examples,
                        epoch_avg_loss=train_epoch_avg_loss, epoch_moving_win_loss=train_epoch_window_loss,
                        validation_loss=val_loss, validation_metrics_results=val_metrics_results,
                        avg_throughput=avg_throughput, learning_rates=learning_rates)

                model.train()
                criterion.train()

            for callback in callbacks:
                callback.step_end(
                    epoch_nr=epoch_nr, step_nr=step_nr, batch_nr=batch_idx + 1, nr_batches_in_epoch=nr_batches_in_epoch,
                    batch_loss=batch_loss, batch_nr_examples=batch_nr_examples, epoch_avg_loss=train_epoch_avg_loss,
                    epoch_moving_win_loss=train_epoch_window_loss, avg_throughput=avg_throughput,
                    learning_rates=learning_rates)

            step_nr += 1

        # TODO: make it a callback of `epoch_end`
        if save_checkpoint_fn is not None:
            save_checkpoint_fn(model, optimizer, epoch_nr, None)

        if valid_loader is not None:
            evaluation_avg_duration = evaluation_scheduler.task_avg_duration
            for callback in callbacks:
                callback.epoch_end_before_evaluation(
                    epoch_nr=epoch_nr, step_nr=step_nr,
                    epoch_avg_loss=train_epoch_avg_loss,
                    epoch_moving_win_loss=train_epoch_window_loss,
                    avg_throughput=avg_throughput, learning_rates=learning_rates)

            print(f'Performing evaluation (over validation) after epoch #{epoch_nr}:')
            val_loss, val_metrics_results = evaluate(
                model=model, device=device, valid_loader=valid_loader, criterion=criterion,
                evaluation_metrics_types=evaluation_metrics_types,
                progress_bar_min_interval_sec=progress_bar_min_interval_sec)
            evaluate_start_time = time.time()
            last_evaluation_duration = time.time() - evaluate_start_time
            evaluation_avg_duration = last_evaluation_duration if evaluation_avg_duration is None else \
                (evaluation_avg_duration * 0.8 + last_evaluation_duration * 0.2)
            # TODO: For pretty printing the evaluation metric results:
            #       https://stackoverflow.com/questions/44356693/pprint-with-custom-float-formats
            print(f'Completed performing training & evaluation for epoch #{epoch_nr}.'
                  f'\n\t validation loss: {val_loss:.4f}'
                  f'\n\t validation metrics: {val_metrics_results}')

            for callback in callbacks:
                callback.epoch_end_after_evaluation(
                    epoch_nr=epoch_nr, step_nr=step_nr,
                    epoch_avg_loss=train_epoch_avg_loss,
                    epoch_moving_win_loss=train_epoch_window_loss,
                    validation_loss=val_loss, validation_metrics_results=val_metrics_results,
                    avg_throughput=avg_throughput, learning_rates=learning_rates)

        for callback in callbacks:
            callback.epoch_end(
                epoch_nr=epoch_nr, step_nr=step_nr,
                epoch_avg_loss=train_epoch_avg_loss,
                epoch_moving_win_loss=train_epoch_window_loss,
                avg_throughput=avg_throughput,
                learning_rates=learning_rates)

        if lr_warmup_scheduler is None or lr_warmup_scheduler.is_finished():
            for lr_scheduler in lr_schedulers:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(train_epoch_avg_loss)
                else:
                    lr_scheduler.step()
        learning_rates = tuple(param_group['lr'] for param_group in optimizer.param_groups)


def evaluate(
        model: nn.Module, device: torch.device, valid_loader: DataLoader, criterion: nn.Module,
        evaluation_metrics_types: List[Type[EvaluationMetric]], lazy_move_to_device_history=None,
        progress_bar_min_interval_sec: float = 0.1) -> Tuple[float, Dict[str, float]]:
    if lazy_move_to_device_history is None:
        lazy_move_to_device_history = {'x': {}, 'y': {}}
    else:
        if 'x' not in lazy_move_to_device_history:
            lazy_move_to_device_history['x'] = {}
        if 'y' not in lazy_move_to_device_history:
            lazy_move_to_device_history['y'] = {}
    model.to(device)
    model.eval()
    criterion.eval()
    metrics = [metric() for metric in evaluation_metrics_types]
    eval_epoch_loss_sum = eval_epoch_nr_examples = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(
                valid_loader, dynamic_ncols=True, position=0, leave=True, mininterval=progress_bar_min_interval_sec):
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

import time
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Callable, List, Type, Tuple, Dict, Collection

from ddfa.nn_utils.window_average import WindowAverage
from ddfa.nn_utils.train_callback import TrainCallback
from ddfa.code_tasks.code_task_base import EvaluationMetric


__all__ = ['fit', 'evaluate']


def perform_loss_step_for_batch(device, x_batch: torch.Tensor, y_batch: torch.Tensor, model: nn.Module,
                                criterion: nn.Module, optimizer: Optional[Optimizer] = None,
                                batch_idx: Optional[int] = None, nr_batches: Optional[int] = None,
                                minibatch_size: Optional[int] = None):
    # torch.cuda.empty_cache()  # this avoids OOM on bigger bsz, but makes all work slowly
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    y_pred = model(x_batch, y_batch)
    loss = criterion(y_pred, y_batch)
    if optimizer is not None:
        loss.backward()
        if minibatch_size is None or (batch_idx % minibatch_size) == minibatch_size - 1 or \
                (nr_batches is not None and batch_idx == nr_batches - 1):
            optimizer.step()
            optimizer.zero_grad()
    return y_pred, loss.item(), x_batch.batch_size


def fit(nr_epochs: int, model: nn.Module, device: torch.device, train_loader: DataLoader,
        valid_loader: Optional[DataLoader], optimizer: Optimizer, criterion: nn.Module = F.nll_loss,
        minibatch_size: Optional[int] = None,
        save_checkpoint_fn: Optional[Callable[[nn.Module, Optimizer, int, Optional[int]], None]] = None,
        evaluation_metrics_types: Optional[List[Type[EvaluationMetric]]] = None,
        callbacks: Optional[Collection[TrainCallback]] = None,
        evaluation_time_consumption_ratio: float = 1/20,
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
    nr_steps_performed_since_last_evaluation = 0
    for epoch_nr in range(1, nr_epochs + 1):
        for callback in callbacks:
            callback.epoch_start(epoch_nr=epoch_nr)
        model.train()
        train_epoch_loss_sum = 0
        train_epoch_nr_examples = 0
        train_epoch_avg_loss = 0.0
        train_epoch_window_loss = WindowAverage(max_window_size=15)
        train_data_loader_with_progress = tqdm(train_loader, dynamic_ncols=True)
        nr_steps = len(train_data_loader_with_progress)
        for batch_idx, (x_batch, y_batch) in enumerate(iter(train_data_loader_with_progress)):
            for callback in callbacks:
                callback.step_start(
                    epoch_nr=epoch_nr, step_nr=batch_idx + 1, nr_steps=nr_steps)
            cur_step_start_time = time.time()
            _, batch_loss, batch_nr_examples = perform_loss_step_for_batch(
                device=device, x_batch=x_batch, y_batch=y_batch, model=model,
                criterion=criterion, optimizer=optimizer, batch_idx=batch_idx,
                nr_batches=nr_steps, minibatch_size=minibatch_size)
            cur_step_duration = time.time() - cur_step_start_time
            train_step_avg_time = cur_step_duration if train_step_avg_time is None else \
                train_step_avg_time * 0.8 + cur_step_duration * 0.2
            nr_steps_performed_since_last_evaluation += 1
            train_epoch_loss_sum += batch_loss * batch_nr_examples
            train_epoch_nr_examples += batch_nr_examples
            train_epoch_window_loss.update(batch_loss)
            train_epoch_avg_loss = train_epoch_loss_sum/train_epoch_nr_examples
            train_data_loader_with_progress.set_postfix(
                {'loss (epoch avg)': f'{train_epoch_avg_loss:.4f}',
                 'loss (win avg)': f'{train_epoch_window_loss.get_window_avg():.4f}',
                 'loss (win stbl avg)': f'{train_epoch_window_loss.get_window_avg_wo_outliers():.4f}'})

            if valid_loader is not None and batch_idx + 1 < nr_steps and \
                    (evaluation_avg_duration / (nr_steps_performed_since_last_evaluation * train_step_avg_time)) < \
                    evaluation_time_consumption_ratio:
                for callback in callbacks:
                    callback.step_end_before_evaluation(
                        epoch_nr=epoch_nr, step_nr=batch_idx + 1, nr_steps=nr_steps,
                        batch_loss=batch_loss, batch_nr_examples=batch_nr_examples,
                        epoch_avg_loss=train_epoch_avg_loss, epoch_moving_win_loss=train_epoch_window_loss)

                print(f'Performing evaluation (over validation) DURING epoch #{epoch_nr} '
                      f'(after step {batch_idx + 1}/{nr_steps}):')
                evaluate_start_time = time.time()
                val_loss, val_metrics_results = evaluate(
                    model=model, device=device, valid_loader=valid_loader, criterion=criterion,
                    evaluation_metrics_types=evaluation_metrics_types)
                last_evaluation_duration = time.time() - evaluate_start_time
                evaluation_avg_duration = last_evaluation_duration if evaluation_avg_duration is None else \
                    (evaluation_avg_duration + last_evaluation_duration) / 2
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

                nr_steps_performed_since_last_evaluation = 0

            for callback in callbacks:
                callback.step_end(
                    epoch_nr=epoch_nr, step_nr=batch_idx + 1, nr_steps=nr_steps,
                    batch_loss=batch_loss, batch_nr_examples=batch_nr_examples, epoch_avg_loss=train_epoch_avg_loss,
                    epoch_moving_win_loss=train_epoch_window_loss)

        # TODO: make it a callback of `epoch_end`
        if save_checkpoint_fn is not None:
            save_checkpoint_fn(model, optimizer, epoch_nr, None)

        if valid_loader is not None:
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
                (evaluation_avg_duration + last_evaluation_duration) / 2
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


def evaluate(model: nn.Module, device: torch.device, valid_loader: DataLoader, criterion: nn.Module,
             evaluation_metrics_types: List[Type[EvaluationMetric]]) -> Tuple[float, Dict[str, float]]:
    model.to(device)
    model.eval()
    metrics = [metric() for metric in evaluation_metrics_types]
    eval_epoch_loss_sum = eval_epoch_nr_examples = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(valid_loader, dynamic_ncols=True):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
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

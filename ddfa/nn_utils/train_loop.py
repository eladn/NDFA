import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Callable, List, Type
import numpy as np

from ddfa.nn_utils.window_average import WindowAverage
from ddfa.code_tasks.code_task_base import EvaluationMetric


__all__ = ['fit']


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
    return y_pred, loss.item(), len(x_batch)


def fit(nr_epochs: int, model: nn.Module, device: torch.device, train_loader: DataLoader,
        valid_loader: Optional[DataLoader], optimizer: Optimizer, criterion: nn.Module = F.nll_loss,
        minibatch_size: Optional[int] = None,
        save_checkpoint_fn: Optional[Callable[[nn.Module, Optimizer, int, Optional[int]], None]] = None,
        evaluation_metrics_types: Optional[List[Type[EvaluationMetric]]] = None):
    model.to(device)
    for epoch_nr in range(1, nr_epochs + 1):
        model.train()
        train_epoch_loss_sum = 0
        train_epoch_nr_examples = 0
        train_epoch_window_loss = WindowAverage(max_window_size=15)
        train_data_loader_with_progress = tqdm(train_loader, dynamic_ncols=True)
        for batch_idx, (x_batch, y_batch) in enumerate(iter(train_data_loader_with_progress)):
            _, batch_loss, batch_nr_examples = perform_loss_step_for_batch(
                device=device, x_batch=x_batch, y_batch=y_batch, model=model,
                criterion=criterion, optimizer=optimizer, batch_idx=batch_idx,
                nr_batches=len(train_data_loader_with_progress), minibatch_size=minibatch_size)
            train_epoch_loss_sum += batch_loss * batch_nr_examples
            train_epoch_nr_examples += batch_nr_examples
            train_epoch_window_loss.update(batch_loss)
            train_data_loader_with_progress.set_postfix(
                {'loss (epoch avg)': f'{train_epoch_loss_sum/train_epoch_nr_examples:.4f}',
                 'loss (win avg)': f'{train_epoch_window_loss.get_window_avg():.4f}',
                 'loss (win stbl avg)': f'{train_epoch_window_loss.get_window_avg_wo_outliers():.4f}'})

        if save_checkpoint_fn is not None:
            save_checkpoint_fn(model, optimizer, epoch_nr)

        if valid_loader is not None:
            print(f'Performing evaluation (over validation) after epoch #{epoch_nr}:')
            model.eval()
            metrics = [metric() for metric in evaluation_metrics_types]
            eval_epoch_loss_sum = eval_epoch_nr_examples = 0
            with torch.no_grad():
                for x_batch, y_batch in tqdm(valid_loader, dynamic_ncols=True):
                    y_hat, batch_loss, batch_nr_examples = perform_loss_step_for_batch(
                        device=device, x_batch=x_batch, y_batch=y_batch, model=model, criterion=criterion)
                    for metric in metrics:
                        metric.update(y_hat=y_hat.item(), target=y_batch)
                    eval_epoch_loss_sum += batch_loss * batch_nr_examples
                    eval_epoch_nr_examples += batch_nr_examples
            val_loss = eval_epoch_loss_sum / eval_epoch_nr_examples
            metrics_results = {}
            for metric in metrics:
                metrics_results.update(metric.get_metrics())
            print(f'Completed performing training & evaluation for epoch #{epoch_nr}.'
                  f'\n\t validation loss: {val_loss:.4f}'
                  f'\n\t validation metrics: {metrics_results}')


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

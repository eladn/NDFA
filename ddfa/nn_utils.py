import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional
import numpy as np


def perform_loss_step_for_batch(device, x_batch: torch.Tensor, y_batch: torch.Tensor, model: nn.Module,
                                optimizer: Optional[Optimizer], criterion: nn.Module):
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    y_pred = model(x_batch)
    loss = criterion(y_pred, y_batch)
    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.item(), len(x_batch)


def fit(nr_epochs: int, model: nn.Module, device: torch.device, train_loader: DataLoader, valid_loader: DataLoader,
        optimizer: Optimizer, criterion: nn.Module = F.nll_loss):
    model.to(device)
    for epoch_nr in range(1, nr_epochs + 1):
        model.train()
        train_epoch_loss_sum = 0
        train_epoch_nr_examples = 0
        train_data_loader_with_progress = tqdm(train_loader)
        for x_batch, y_batch in iter(train_data_loader_with_progress):
            batch_loss, batch_nr_examples = perform_loss_step_for_batch(
                device, x_batch, y_batch, model, optimizer, criterion)
            train_epoch_loss_sum += batch_loss * batch_nr_examples
            train_epoch_nr_examples += batch_nr_examples
            train_data_loader_with_progress.set_postfix(f'avg loss:{train_epoch_loss_sum/train_epoch_nr_examples:.4f}')

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *(perform_loss_step_for_batch(device, x_batch, y_batch, model, optimizer, criterion)
                  for x_batch, y_batch in valid_loader))
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(f'Train Epoch #{epoch_nr} -- validation loss: {val_loss:.4f}')


def train_epoch(args, epoch_nr: int, model: nn.Module, device, train_loader: DataLoader, optimizer: Optimizer, criterion: nn.Module = F.nll_loss):
    model.train()
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_nr, batch_idx * len(x_batch), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


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


def apply_batched_embeddings(embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    assert len(embeddings.size()) == 3  # (batch_size, nr_words_per_example, embedding_dim)
    assert len(indices.size()) >= 1
    assert embeddings.size()[0] == indices.size()[0]  # same batch_size
    batch_size, nr_words_per_example, embedding_dim = embeddings.size()
    indices_non_batch_dims = indices.size()[1:]
    nr_indices_per_example = min(1, np.product(indices_non_batch_dims))
    indices_flattened = indices.flatten()  # (batch_size * nr_indices_per_example,)
    assert indices_flattened.size()[0] == batch_size * nr_indices_per_example
    indices_offsets_fixes = (torch.range(start=0, end=batch_size-1) * nr_words_per_example)\
        .repeat((nr_indices_per_example, 1)).T.flatten()  #  = [0,0,...0,1,1,...,1, ...]
    assert indices_flattened.size() == indices_offsets_fixes.size()
    indices_flattened_with_fixed_offsets = indices_flattened + indices_offsets_fixes
    embeddings_flattened = embeddings.flatten(0, 1)  # (batch_size * nr_words_per_example, embedding_dim)
    selected_embedding_vectors_flattened = embeddings_flattened[indices_flattened_with_fixed_offsets]  # (batch_size * nr_indices_per_example, embedding_dim)
    assert selected_embedding_vectors_flattened.size() == (
        batch_size * nr_indices_per_example, embedding_dim)
    selected_embedding_vectors = selected_embedding_vectors_flattened.view(indices.size() + (embedding_dim,))
    return selected_embedding_vectors

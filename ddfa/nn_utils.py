import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Union
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


def apply_batched_embeddings(
        batched_embeddings: torch.Tensor, indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None, padding_embedding_vector: Optional[torch.Tensor] = None,
        common_embeddings: Optional[Union[torch.Tensor, nn.Embedding]] = None) -> torch.Tensor:
    assert len(batched_embeddings.size()) == 3  # (batch_size, nr_words_per_example, embedding_dim)
    assert indices.dtype in {torch.int, torch.int32, torch.int64, torch.long}
    assert len(indices.size()) >= 1
    assert batched_embeddings.size()[0] == indices.size()[0]  # same batch_size
    batch_size, nr_words_per_example, embedding_dim = batched_embeddings.size()
    assert common_embeddings is None or \
           (len(common_embeddings.size()) == 2 and common_embeddings.size()[1] == embedding_dim)
    assert (mask is None) ^ (padding_embedding_vector is not None)
    assert padding_embedding_vector is None or \
           (len(padding_embedding_vector.size()) == 1 and padding_embedding_vector.size()[0] == embedding_dim)
    assert mask is None or mask.size() == indices.size()
    if common_embeddings is not None and mask is not None:
        raise ValueError('Can specify either `common_embeddings` or `mask`, but not both.')
    indices_non_batch_dims = indices.size()[1:]
    nr_indices_per_example = max(1, np.product(indices_non_batch_dims))
    indices_flattened = indices.flatten()  # (batch_size * nr_indices_per_example,)
    assert indices_flattened.size() == (batch_size * nr_indices_per_example,)
    indices_offsets_fixes = (torch.arange(batch_size) * nr_words_per_example)\
        .repeat((nr_indices_per_example, 1)).T.flatten().type_as(indices)  #  = [0,0,...,0,1,1,...,1, ...]
    assert indices_flattened.size() == indices_offsets_fixes.size()
    embeddings_flattened = batched_embeddings.flatten(0, 1)  # (batch_size * nr_words_per_example, embedding_dim)
    if common_embeddings is None and mask is None:
        indices_flattened_with_fixed_offsets = indices_flattened + indices_offsets_fixes
        selected_embedding_vectors_flattened = embeddings_flattened[
            indices_flattened_with_fixed_offsets.type(dtype=torch.long)]  # (batch_size * nr_indices_per_example, embedding_dim)
    elif mask is not None:
        mask_flattened = mask.flatten().view(-1, 1)  # (batch_size * nr_indices_per_example, 1)
        indices_flattened_with_fixed_offsets = torch.where(
            mask_flattened,
            indices_flattened + indices_offsets_fixes,
            torch.zeros(indices_flattened.size()))
        selected_embedding_vectors_flattened = torch.where(
            mask_flattened,
            embeddings_flattened[indices_flattened_with_fixed_offsets],
            padding_embedding_vector)  # (batch_size * nr_indices_per_example, embedding_dim)
    else:  # common_embeddings is not None
        nr_common_embeddings = int(common_embeddings.size()[0])
        use_common_embeddings_mask = (indices_flattened < nr_common_embeddings)
        indices_flattened_with_fixed_offsets = torch.where(
            use_common_embeddings_mask,
            indices_flattened,
            indices_flattened + indices_offsets_fixes - nr_common_embeddings)
        embeddings_indices = torch.where(
                use_common_embeddings_mask,
                torch.zeros(indices_flattened_with_fixed_offsets.size(), dtype=torch.long),  # avoid accessing invalid index
                indices_flattened_with_fixed_offsets.type(dtype=torch.long))
        common_embeddings_indices = torch.where(
                use_common_embeddings_mask,
                indices_flattened_with_fixed_offsets.type(dtype=torch.long),
                torch.zeros(indices_flattened_with_fixed_offsets.size(), dtype=torch.long))  # avoid accessing invalid index
        applied_common_embeddings = common_embeddings(common_embeddings_indices) \
            if isinstance(common_embeddings, nn.Embedding) else common_embeddings[common_embeddings_indices]
        selected_embedding_vectors_flattened = torch.where(
            use_common_embeddings_mask.view(-1, 1),
            applied_common_embeddings,
            embeddings_flattened[embeddings_indices])
    assert selected_embedding_vectors_flattened.size() == (
        batch_size * nr_indices_per_example, embedding_dim)
    selected_embedding_vectors = selected_embedding_vectors_flattened.view(indices.size() + (embedding_dim,))
    return selected_embedding_vectors


def apply_batched_embeddings_test():
    batched_embeddings = torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # 3 words of example #1
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]],  # 3 words of example #2
        [[100, 200, 300], [400, 500, 600], [700, 800, 900]],  # 3 words of example #3
        [[1000, 2000, 3000], [4000, 5000, 6000], [7000, 8000, 9000]],  # 3 words of example #4
    ])  # (batch_size, nr_words_per_example, embedding_dim)
    common_embeddings = torch.tensor([
        [11, 22, 33], [44, 55, 66], [77, 88, 99], [111, 222, 333], [444, 555, 666]  # 5 common words
    ])  # (nr_common_words, embedding_dim)
    indices_multiple_seqs_per_example = torch.tensor([
        [[1, 2, 0, 1], [2, 0, 0, 1]],  # 2 seqs for example #1 in batch
        [[2, 1, 0, 2], [1, 1, 2, 0]],  # 2 seqs for example #2 in batch
        [[0, 2, 1, 1], [2, 1, 0, 0]],  # 2 seqs for example #3 in batch
        [[0, 0, 1, 2], [2, 2, 1, 0]],  # 2 seqs for example #4 in batch
    ], dtype=torch.int)  # (batch_size, nr_seqs_per_example, seq_len)

    wo_common_wo_mask_expected_result = torch.tensor([
        [
            [[4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [1, 2, 3], [1, 2, 3], [4, 5, 6]]
        ],  # 2 embedded seqs for example #1 in batch
        [
            [[70, 80, 90], [40, 50, 60], [10, 20, 30], [70, 80, 90]],
            [[40, 50, 60], [40, 50, 60], [70, 80, 90], [10, 20, 30]]
        ],  # 2 embedded seqs for example #2 in batch
        [
            [[100, 200, 300], [700, 800, 900], [400, 500, 600], [400, 500, 600]],
            [[700, 800, 900], [400, 500, 600], [100, 200, 300], [100, 200, 300]]
        ],  # 2 embedded seqs for example #3 in batch
        [
            [[1000, 2000, 3000], [1000, 2000, 3000], [4000, 5000, 6000], [7000, 8000, 9000]],
            [[7000, 8000, 9000], [7000, 8000, 9000], [4000, 5000, 6000], [1000, 2000, 3000]]
        ],  # 2 embedded seqs for example #4 in batch
    ])  # (batch_size, nr_seqs_per_example, seq_len, embedding_dim)
    applied_embd = apply_batched_embeddings(
        batched_embeddings=batched_embeddings,
        indices=indices_multiple_seqs_per_example)
    assert applied_embd.size() == wo_common_wo_mask_expected_result.size()
    assert torch.all(applied_embd == wo_common_wo_mask_expected_result)

    # test for the case of a single sequence per example.
    for seq_idx in range(wo_common_wo_mask_expected_result.size()[1]):
        applied_embd = apply_batched_embeddings(
            batched_embeddings=batched_embeddings,
            indices=indices_multiple_seqs_per_example[:, seq_idx, :])
        assert applied_embd.size() == wo_common_wo_mask_expected_result[:, seq_idx, :, :].size()
        assert torch.all(applied_embd == wo_common_wo_mask_expected_result[:, seq_idx, :, :])

    indices_multiple_seqs_per_example_with_nonused_common_embeddings = indices_multiple_seqs_per_example + 5
    with_nonused_common_wo_mask_expected_result = wo_common_wo_mask_expected_result  # (batch_size, nr_seqs_per_example, seq_len, embedding_dim)
    applied_embd = apply_batched_embeddings(
        batched_embeddings=batched_embeddings,
        indices=indices_multiple_seqs_per_example_with_nonused_common_embeddings,
        common_embeddings=common_embeddings)
    assert applied_embd.size() == with_nonused_common_wo_mask_expected_result.size()
    assert torch.all(applied_embd == with_nonused_common_wo_mask_expected_result)

    indices_multiple_seqs_per_example_with_used_common_embeddings = indices_multiple_seqs_per_example + 5
    indices_multiple_seqs_per_example_with_used_common_embeddings[0, 1, 2] = 4
    indices_multiple_seqs_per_example_with_used_common_embeddings[1, 0, 1] = 0
    indices_multiple_seqs_per_example_with_used_common_embeddings[2, 1, 1] = 3
    indices_multiple_seqs_per_example_with_used_common_embeddings[2, 0, 3] = 2
    indices_multiple_seqs_per_example_with_used_common_embeddings[3, 0, 0] = 1
    with_used_common_wo_mask_expected_result = wo_common_wo_mask_expected_result  # (batch_size, nr_seqs_per_example, seq_len, embedding_dim)
    with_used_common_wo_mask_expected_result[0, 1, 2, :] = torch.tensor([444, 555, 666])
    with_used_common_wo_mask_expected_result[1, 0, 1, :] = torch.tensor([11, 22, 33])
    with_used_common_wo_mask_expected_result[2, 1, 1, :] = torch.tensor([111, 222, 333])
    with_used_common_wo_mask_expected_result[2, 0, 3, :] = torch.tensor([77, 88, 99])
    with_used_common_wo_mask_expected_result[3, 0, 0, :] = torch.tensor([44, 55, 66])
    applied_embd = apply_batched_embeddings(
        batched_embeddings=batched_embeddings,
        indices=indices_multiple_seqs_per_example_with_used_common_embeddings,
        common_embeddings=common_embeddings)
    assert applied_embd.size() == with_used_common_wo_mask_expected_result.size()
    assert torch.all(applied_embd == with_used_common_wo_mask_expected_result)

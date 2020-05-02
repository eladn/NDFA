import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Union, Callable
import numpy as np


def perform_loss_step_for_batch(device, x_batch: torch.Tensor, y_batch: torch.Tensor, model: nn.Module,
                                criterion: nn.Module, optimizer: Optional[Optimizer] = None):
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    y_pred = model(x_batch, y_batch)
    loss = criterion(y_pred, y_batch)
    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.item(), len(x_batch)


class CyclicAppendOnlyBuffer:
    def __init__(self, buffer_size: int):
        assert buffer_size > 0
        self._buffer_size = buffer_size
        self._nr_items = 0
        self._buffer = [None] * buffer_size
        self._next_insert_idx = 0

    def insert(self, item):
        self._buffer[self._next_insert_idx] = item
        self._next_insert_idx = (self._next_insert_idx + 1) % self._buffer_size
        self._nr_items = min(self._nr_items + 1, self._buffer_size)

    def get_all_items(self):
        return (self._buffer[self._next_insert_idx:] if self._nr_items == self._buffer_size else []) + \
               self._buffer[:self._next_insert_idx]

    def __len__(self) -> int:
        return self._nr_items


class WindowAverage:
    def __init__(self, max_window_size: int = 5):
        self._max_window_size = max_window_size
        self._items_buffer = CyclicAppendOnlyBuffer(buffer_size=max_window_size)

    def update(self, number: float):
        self._items_buffer.insert(number)

    def get_window_avg(self, sub_window_size: Optional[int] = None):
        assert sub_window_size is None or sub_window_size <= self._max_window_size
        window_size = self._max_window_size if sub_window_size is None else sub_window_size
        return np.average(self._items_buffer.get_all_items()[:window_size])

    def get_window_avg_wo_outliers(self, nr_outliers: int = 3, sub_window_size: Optional[int] = None):
        assert sub_window_size is None or sub_window_size <= self._max_window_size
        window_size = self._max_window_size if sub_window_size is None else sub_window_size
        assert nr_outliers < window_size
        nr_outliers_wrt_windows_size_max_ratio = nr_outliers / window_size
        assert nr_outliers_wrt_windows_size_max_ratio < 1
        eff_window_size = min(len(self._items_buffer), window_size)
        if eff_window_size < window_size and nr_outliers >= eff_window_size * nr_outliers_wrt_windows_size_max_ratio:
            nr_outliers = int(eff_window_size * nr_outliers_wrt_windows_size_max_ratio)
        assert nr_outliers <= eff_window_size * nr_outliers_wrt_windows_size_max_ratio
        items = np.array(self._items_buffer.get_all_items()[:eff_window_size])
        if nr_outliers > 0:
            median = np.median(items)
            dist_from_median = np.abs(items - median)
            outliers_indices = np.argpartition(dist_from_median, -nr_outliers)[-nr_outliers:]
            items[outliers_indices] = np.nan
        return np.nanmean(items)


def fit(nr_epochs: int, model: nn.Module, device: torch.device, train_loader: DataLoader,
        valid_loader: Optional[DataLoader], optimizer: Optimizer, criterion: nn.Module = F.nll_loss,
        save_checkpoint_fn: Optional[Callable[[nn.Module, Optimizer, int, Optional[int]], None]] = None):
    model.to(device)
    for epoch_nr in range(1, nr_epochs + 1):
        model.train()
        train_epoch_loss_sum = 0
        train_epoch_nr_examples = 0
        train_epoch_window_loss = WindowAverage(max_window_size=15)
        train_data_loader_with_progress = tqdm(train_loader, dynamic_ncols=True)
        for x_batch, y_batch in iter(train_data_loader_with_progress):
            batch_loss, batch_nr_examples = perform_loss_step_for_batch(
                device=device, x_batch=x_batch, y_batch=y_batch, model=model,
                criterion=criterion, optimizer=optimizer)
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
            model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *(perform_loss_step_for_batch(
                        device=device, x_batch=x_batch, y_batch=y_batch, model=model, criterion=criterion)
                      for x_batch, y_batch in tqdm(valid_loader, dynamic_ncols=True)))
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            print(f'Train Epoch #{epoch_nr} -- validation loss: {val_loss:.4f}')


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
    indices_device = indices.device
    assert len(batched_embeddings.size()) == 3  # (batch_size, nr_words_per_example, embedding_dim)
    assert indices.dtype in {torch.int, torch.int32, torch.int64, torch.long}
    assert len(indices.size()) >= 1
    assert batched_embeddings.size()[0] == indices.size()[0]  # same batch_size
    batch_size, nr_words_per_example, embedding_dim = batched_embeddings.size()
    assert common_embeddings is None or \
           (isinstance(common_embeddings, torch.Tensor) and
            len(common_embeddings.size()) == 2 and common_embeddings.size()[1] == embedding_dim) or \
           (isinstance(common_embeddings, nn.Embedding) and common_embeddings.embedding_dim == embedding_dim)
    assert (mask is None) ^ (padding_embedding_vector is not None)
    assert padding_embedding_vector is None or \
           (len(padding_embedding_vector.size()) == 1 and padding_embedding_vector.size()[0] == embedding_dim)
    assert mask is None or mask.size() == indices.size()
    assert mask is None or mask.dtype == torch.bool
    if common_embeddings is not None and mask is not None:
        raise ValueError('Can specify either `common_embeddings` or `mask`, but not both.')
    indices_non_batch_dims = indices.size()[1:]
    nr_indices_per_example = max(1, np.product(indices_non_batch_dims))
    indices_flattened = indices.flatten().type(dtype=torch.long)  # (batch_size * nr_indices_per_example,)
    assert indices_flattened.size() == (batch_size * nr_indices_per_example,)
    indices_offsets_fixes = (torch.arange(batch_size, dtype=torch.long, device=indices_device) * nr_words_per_example)\
        .repeat((nr_indices_per_example, 1)).T.flatten().type_as(indices)  #  = [0,0,...,0,1,1,...,1, ...]  # TODO: can we use `expand()` instead of `repeat()` here? it uses less memory.
    assert indices_flattened.size() == indices_offsets_fixes.size()
    embeddings_flattened = batched_embeddings.flatten(0, 1)  # (batch_size * nr_words_per_example, embedding_dim)
    if common_embeddings is None and mask is None:
        indices_flattened_with_fixed_offsets = indices_flattened + indices_offsets_fixes
        selected_embedding_vectors_flattened = embeddings_flattened[
            indices_flattened_with_fixed_offsets.type(dtype=torch.long)]  # (batch_size * nr_indices_per_example, embedding_dim)
    elif mask is not None:
        mask_flattened = mask.flatten()  # (batch_size * nr_indices_per_example, 1)
        assert mask_flattened.size() == indices_flattened.size() == indices_offsets_fixes.size()
        indices_flattened_with_fixed_offsets = torch.where(
            mask_flattened,
            indices_flattened + indices_offsets_fixes,
            torch.zeros_like(indices_flattened)).view(batch_size * nr_indices_per_example)
        selected_embedding_vectors_flattened = torch.where(
            mask_flattened.view(-1, 1),
            embeddings_flattened[indices_flattened_with_fixed_offsets],
            padding_embedding_vector)  # (batch_size * nr_indices_per_example, embedding_dim)
    else:  # common_embeddings is not None
        nr_common_embeddings = common_embeddings.num_embeddings if isinstance(common_embeddings, nn.Embedding) else \
            int(common_embeddings.size()[0])
        use_common_embeddings_mask = (indices_flattened < nr_common_embeddings)
        indices_flattened_with_fixed_offsets = torch.where(
            use_common_embeddings_mask,
            indices_flattened,
            indices_flattened + indices_offsets_fixes - nr_common_embeddings)
        embeddings_indices = torch.where(
                use_common_embeddings_mask,
                torch.zeros_like(indices_flattened_with_fixed_offsets),  # avoid accessing invalid index
                indices_flattened_with_fixed_offsets)
        common_embeddings_indices = torch.where(
                use_common_embeddings_mask,
                indices_flattened_with_fixed_offsets,
                torch.zeros_like(indices_flattened_with_fixed_offsets))  # avoid accessing invalid index
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

    wo_common_with_mask_expected_result = wo_common_wo_mask_expected_result.clone()  # (batch_size, nr_seqs_per_example, seq_len, embedding_dim)
    padding_embedding_vector = torch.tensor([666, 777, 888])
    wo_common_with_mask_expected_result[0, 1, 2] = padding_embedding_vector
    wo_common_with_mask_expected_result[1, 0, 1] = padding_embedding_vector
    wo_common_with_mask_expected_result[2, 1, 1] = padding_embedding_vector
    wo_common_with_mask_expected_result[2, 0, 3] = padding_embedding_vector
    wo_common_with_mask_expected_result[3, 0, 0] = padding_embedding_vector
    mask = torch.ones(indices_multiple_seqs_per_example.size()).type(torch.bool)
    mask[0, 1, 2] = False
    mask[1, 0, 1] = False
    mask[2, 1, 1] = False
    mask[2, 0, 3] = False
    mask[3, 0, 0] = False
    applied_embd = apply_batched_embeddings(
        batched_embeddings=batched_embeddings,
        indices=indices_multiple_seqs_per_example,
        mask=mask, padding_embedding_vector=padding_embedding_vector)
    assert applied_embd.size() == wo_common_with_mask_expected_result.size()
    assert torch.all(applied_embd == wo_common_with_mask_expected_result)

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
    common_embeddings_as_embedding_layer = nn.Embedding(
        num_embeddings=common_embeddings.size()[0],
        embedding_dim=common_embeddings.size()[1],
        _weight=common_embeddings.float())
    applied_embd = apply_batched_embeddings(
        batched_embeddings=batched_embeddings.float(),
        indices=indices_multiple_seqs_per_example_with_used_common_embeddings,
        common_embeddings=common_embeddings_as_embedding_layer)
    assert applied_embd.size() == with_used_common_wo_mask_expected_result.size()
    assert torch.all(applied_embd.isclose(with_used_common_wo_mask_expected_result.float()))

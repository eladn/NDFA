import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union


__all__ = ['apply_batched_embeddings', 'apply_batched_flattened_embeddings']


def apply_batched_embeddings(
        batched_embeddings: torch.Tensor, indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None, padding_embedding_vector: Optional[torch.Tensor] = None,
        common_embeddings: Optional[Union[torch.Tensor, nn.Embedding]] = None) -> torch.Tensor:
    indices_device = indices.device
    assert batched_embeddings.ndim == 3  # (batch_size, nr_words_per_example, embedding_dim)
    assert indices.dtype in {torch.int, torch.int32, torch.int64, torch.long}
    assert indices.ndim >= 1
    assert batched_embeddings.size(0) == indices.size(0)  # same batch_size
    batch_size, nr_words_per_example, embedding_dim = batched_embeddings.size()
    assert common_embeddings is None or \
           (isinstance(common_embeddings, torch.Tensor) and
            common_embeddings.ndim == 2 and common_embeddings.size(1) == embedding_dim) or \
           (isinstance(common_embeddings, nn.Embedding) and common_embeddings.embedding_dim == embedding_dim)
    assert (mask is None) ^ (padding_embedding_vector is not None)
    assert padding_embedding_vector is None or \
           (padding_embedding_vector.ndim == 1 and padding_embedding_vector.size(0) == embedding_dim)
    assert mask is None or mask.size() == indices.size()
    assert mask is None or mask.dtype == torch.bool
    if common_embeddings is not None and mask is not None:
        raise ValueError('Can specify either `common_embeddings` or `mask`, but not both.')
    indices_non_batch_dims = indices.size()[1:]
    nr_indices_per_example = max(1, np.product(indices_non_batch_dims))
    indices_flattened = indices.flatten().type(dtype=torch.long)  # (batch_size * nr_indices_per_example,)
    assert indices_flattened.size() == (batch_size * nr_indices_per_example,)
    indices_offsets_fixes = (torch.arange(batch_size, dtype=torch.long, device=indices_device) * nr_words_per_example)\
        .unsqueeze(0).expand(nr_indices_per_example, -1).T.flatten().type_as(indices)  #  = [0,0,...,0,1,1,...,1, ...]
    assert indices_flattened.size() == indices_offsets_fixes.size()
    embeddings_flattened = batched_embeddings.flatten(0, 1)  # (batch_size * nr_words_per_example, embedding_dim)
    zeros_tensor = torch.zeros(1, dtype=torch.long, device=indices_device)
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
            zeros_tensor  # torch.zeros_like(indices_flattened)
        ).view(batch_size * nr_indices_per_example)
        selected_embedding_vectors_flattened = torch.where(
            mask_flattened.view(-1, 1),
            embeddings_flattened[indices_flattened_with_fixed_offsets],
            padding_embedding_vector)  # (batch_size * nr_indices_per_example, embedding_dim)
    else:  # common_embeddings is not None
        nr_common_embeddings = common_embeddings.num_embeddings if isinstance(common_embeddings, nn.Embedding) else \
            int(common_embeddings.size(0))
        use_common_embeddings_mask = (indices_flattened < nr_common_embeddings)
        indices_flattened_with_fixed_offsets = torch.where(
            use_common_embeddings_mask,
            indices_flattened,
            indices_flattened + indices_offsets_fixes - nr_common_embeddings)
        embeddings_indices = torch.where(
                use_common_embeddings_mask,
                zeros_tensor,  # torch.zeros_like(indices_flattened_with_fixed_offsets)  #avoid accessing invalid index
                indices_flattened_with_fixed_offsets)
        common_embeddings_indices = torch.where(
                use_common_embeddings_mask,
                indices_flattened_with_fixed_offsets,
                zeros_tensor)  # torch.zeros_like(indices_flattened_with_fixed_offsets)  # avoid accessing invalid index
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


def apply_batched_flattened_embeddings(
        indices: torch.LongTensor,
        batched_flattened_encodings: torch.Tensor,
        common_embeddings: Optional[Union[torch.Tensor, nn.Embedding]] = None) -> torch.Tensor:
    assert batched_flattened_encodings.ndim == 2
    nr_common_embeddings = 0 if common_embeddings is None else \
        common_embeddings.num_embeddings if isinstance(common_embeddings, nn.Embedding) else \
        int(common_embeddings.size(0))
    if nr_common_embeddings < 1:
        return batched_flattened_encodings[indices]
    is_common_idx_cond = indices < indices.new_full(size=(1,), fill_value=nr_common_embeddings)
    common_indices = torch.where(is_common_idx_cond, indices, indices.new_zeros(1))
    applied_common_encodings = common_embeddings(common_indices) if isinstance(common_embeddings, nn.Embedding) else \
        common_embeddings[common_indices]
    batch_flattened_embeddings_indices = torch.where(
        is_common_idx_cond, indices.new_zeros(1), indices - nr_common_embeddings)
    applied_batched_encodings = batched_flattened_encodings[batch_flattened_embeddings_indices]
    return torch.where(
        is_common_idx_cond.unsqueeze(-1).expand(applied_common_encodings.size()),
        applied_common_encodings, applied_batched_encodings)


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
    for seq_idx in range(wo_common_wo_mask_expected_result.size(1)):
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
        num_embeddings=common_embeddings.size(0),
        embedding_dim=common_embeddings.size(1),
        _weight=common_embeddings.float())
    applied_embd = apply_batched_embeddings(
        batched_embeddings=batched_embeddings.float(),
        indices=indices_multiple_seqs_per_example_with_used_common_embeddings,
        common_embeddings=common_embeddings_as_embedding_layer)
    assert applied_embd.size() == with_used_common_wo_mask_expected_result.size()
    assert torch.all(applied_embd.isclose(with_used_common_wo_mask_expected_result.float()))


if __name__ == '__main__':
    apply_batched_embeddings_test()

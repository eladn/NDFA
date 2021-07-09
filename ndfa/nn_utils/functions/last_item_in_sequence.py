import torch


__all__ = ['get_last_item_in_sequence']


def get_last_item_in_sequence(sequence_encodings: torch.Tensor, sequence_lengths: torch.LongTensor):
    assert len(sequence_encodings.shape) == 3
    assert sequence_encodings.shape[:1] == sequence_lengths.shape
    last_word_indices = (sequence_lengths - 1).view(sequence_encodings.size(0), 1, 1) \
        .expand(sequence_encodings.size(0), 1, sequence_encodings.size(2))
    last_word_encoding = torch.gather(
        sequence_encodings, dim=1, index=last_word_indices).squeeze(1)
    return last_word_encoding

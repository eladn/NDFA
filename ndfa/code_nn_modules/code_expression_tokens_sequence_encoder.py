__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2020-11-29"

from functools import reduce
from typing import Optional, Collection, Tuple

import torch
import torch.nn as nn

from ndfa.code_nn_modules.params.code_tokens_seq_encoder_params import CodeTokensSeqEncoderParams
from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.nn_utils.modules.sequence_encoder import SequenceEncoder
from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams
from ndfa.nn_utils.modules.norm_wrapper import NormWrapper
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.misc.code_data_structure_api import SerTokenKind


__all__ = ['CodeExpressionTokensSequenceEncoder']


def _create_tokens_mask_by_kind(
        tokens_kinds_vocab: Vocabulary,
        token_kinds: Collection[str],
        expressions_input: CodeExpressionTokensSequenceInputTensors) -> torch.BoolTensor:
    token_kinds = set(token_kinds)
    # kos_token_kinds = {SerTokenKind.KEYWORD.value, SerTokenKind.OPERATOR.value, SerTokenKind.SEPARATOR.value}
    all_token_kinds = {item.value for item in SerTokenKind.__members__.values()}
    if token_kinds & all_token_kinds != token_kinds:
        raise ValueError(f'Unsupported token kinds `{token_kinds - all_token_kinds}`.')
    token_kinds_indices = [tokens_kinds_vocab.get_word_idx(token_kind) for token_kind in token_kinds]
    tokens_masks = [expressions_input.token_type.sequences == token_kind_idx for token_kind_idx in token_kinds_indices]
    return reduce(torch.logical_or, tokens_masks)


def _filter_out_entries_from_sequence_batch_by_mask(sequences: torch.Tensor, mask: torch.BoolTensor) \
        -> Tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
    assert sequences.ndim == 3 and mask.ndim == 2
    assert sequences.shape[:-1] == mask.shape
    indices = torch.masked_fill(torch.cumsum(mask.int(), dim=1), ~mask, 0)
    sequence_lengths = torch.max(indices, dim=1).values.long()
    masked = torch.scatter(
        input=sequences.new_zeros(size=(sequences.size(0), sequences.size(1) + 1, sequences.size(2))),
        dim=1, index=indices.unsqueeze(-1).expand(sequences.shape), src=sequences)[:, 1:, :]
    return masked, sequence_lengths, torch.masked_fill(indices - 1, ~mask, 0)


"""
Usage example for `_filter_out_entries_from_sequence_batch_by_mask()` and its inverse via `torch.gather()`.
>>> a
tensor([[[ 1],
         [ 2],
         [ 3],
         [ 4]],
        [[10],
         [11],
         [12],
         [13]]])
>>> m
tensor([[ True, False, False,  True],
        [False,  True, False, False]])
>>> masked, lengths, indices = _filter_out_entries_from_sequence_batch_by_mask(sequences=a, mask=m)
>>> masked
tensor([[[ 1],
         [ 4],
         [ 0],
         [ 0]],
        [[11],
         [ 0],
         [ 0],
         [ 0]]])
>>> lengths
tensor([2, 1])
>>> indices
tensor([[0, 0, 0, 1],
        [0, 0, 0, 0]])
>>> processed_masked = masked * 100
>>> processed_masked
tensor([[[ 100],
         [ 400],
         [   0],
         [   0]],
        [[1100],
         [   0],
         [   0],
         [   0]]])
>>> gathered=torch.gather(input=processed_masked, dim=1, index=indices.unsqueeze(-1).expand(masked.shape))
>>> gathered
tensor([[[ 100],
         [ 100],
         [ 100],
         [ 400]],
        [[1100],
         [1100],
         [1100],
         [1100]]])
>>> torch.where(m.unsqueeze(-1).expand(processed_masked.shape), gathered, a)
tensor([[[ 100],
         [   2],
         [   3],
         [ 400]],
        [[  10],
         [1100],
         [  12],
         [  13]]])
"""


class CodeExpressionTokensSequenceEncoder(nn.Module):
    def __init__(
            self,
            encoder_params: CodeTokensSeqEncoderParams,
            tokens_kinds_vocab: Vocabulary,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CodeExpressionTokensSequenceEncoder, self).__init__()
        self.encoder_params = encoder_params
        if self.encoder_params.shuffle_expressions and self.encoder_params.ignore_token_kinds:
            raise ValueError('Cannot use both `shuffle_expressions` together with `ignore_token_kinds`.')
        self.sequence_encoder = SequenceEncoder(
            encoder_params=self.encoder_params.sequence_encoder,
            input_dim=self.encoder_params.token_encoding_dim,
            hidden_dim=self.encoder_params.token_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.tokens_kinds_vocab = tokens_kinds_vocab
        self.norm = None if norm_params is None else NormWrapper(
            nr_features=self.encoder_params.token_encoding_dim, params=norm_params)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self,
                token_seqs_embeddings: torch.Tensor,
                expressions_input: CodeExpressionTokensSequenceInputTensors) \
            -> CodeExpressionEncodingsTensors:
        input_sequences = token_seqs_embeddings
        sequences_lengths = expressions_input.token_type.sequences_lengths
        if self.encoder_params.ignore_token_kinds:
            assert not self.encoder_params.shuffle_expressions
            tokens_to_ignore_mask = _create_tokens_mask_by_kind(
                tokens_kinds_vocab=self.tokens_kinds_vocab,
                token_kinds=self.encoder_params.ignore_token_kinds,
                expressions_input=expressions_input)
            tokens_to_keep_mask = ~tokens_to_ignore_mask
            input_sequences, sequences_lengths, indices_mappings = _filter_out_entries_from_sequence_batch_by_mask(
                sequences=token_seqs_embeddings, mask=tokens_to_keep_mask)

        if self.encoder_params.shuffle_expressions:
            input_sequences = expressions_input.sequence_shuffler.shuffle(token_seqs_embeddings)

        expressions_encodings = self.sequence_encoder(
            sequence_input=input_sequences,
            lengths=sequences_lengths,
            batch_first=True).sequence
        if self.norm:
            expressions_encodings = self.norm(expressions_encodings)

        if self.encoder_params.shuffle_expressions:
            expressions_encodings = expressions_input.sequence_shuffler.unshuffle(expressions_encodings)

        if self.encoder_params.ignore_token_kinds:
            expressions_encodings = torch.gather(
                input=expressions_encodings, dim=1,
                index=indices_mappings.unsqueeze(-1).expand(expressions_encodings.shape))
            assert expressions_encodings.shape == token_seqs_embeddings.shape
            expressions_encodings = torch.where(
                tokens_to_ignore_mask.unsqueeze(-1).expand(expressions_encodings.shape),
                token_seqs_embeddings, expressions_encodings)

        return CodeExpressionEncodingsTensors(token_seqs=expressions_encodings)

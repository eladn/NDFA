import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.nn_utils.functions.apply_batched_embeddings import apply_batched_embeddings, apply_batched_flattened_embeddings
from ndfa.nn_utils.modules.attention import Attention


__all__ = ['AttnRNNDecoder', 'ScatteredEncodings']


@dataclasses.dataclass
class ScatteredEncodings:
    encodings: torch.FloatTensor
    indices: torch.LongTensor
    mask: torch.BoolTensor


def apply_embeddings(
        indices: torch.LongTensor,
        common_embeddings: Optional[Union[torch.Tensor, nn.Embedding]] = None,
        batched_flattened_encodings: Optional[torch.Tensor] = None,
        batched_encodings: Optional[torch.Tensor] = None) -> torch.Tensor:
    if common_embeddings is not None and batched_encodings is None and batched_flattened_encodings is None:
        return common_embeddings(indices) if isinstance(common_embeddings, nn.Embedding) else common_embeddings[indices]
    if batched_flattened_encodings is not None and batched_encodings is None:
        return apply_batched_flattened_embeddings(
            indices=indices, batched_flattened_encodings=batched_flattened_encodings,
            common_embeddings=common_embeddings)
    if batched_encodings is not None and batched_flattened_encodings is None:
        return apply_batched_embeddings(
            batched_embeddings=batched_encodings, indices=indices,
            common_embeddings=common_embeddings)  # (batch_size, target_seq_len, decoder_output_dim)
    raise ValueError(
        'Cannot specify both `batched_flattened_encodings` and `batched_encodings`. '
        'Must specify at least one of `common_embeddings` or `batched_flattened_encodings` or `batched_encodings`.')


class AttnRNNDecoder(nn.Module):
    def __init__(self, encoder_output_dim: int, decoder_hidden_dim: int,
                 decoder_output_dim: int, max_target_seq_len: int,
                 embedding_dropout_rate: Optional[float] = 0.3, activation_fn: str = 'relu',
                 rnn_type: str = 'lstm', nr_rnn_layers: int = 1,
                 output_common_embedding: Optional[Union[torch.Tensor, nn.Embedding]] = None,
                 output_common_vocab: Optional[Vocabulary] = None):
        assert rnn_type in {'lstm', 'gru'}
        super(AttnRNNDecoder, self).__init__()
        self.activation_layer = get_activation_layer(activation_fn)()
        self.encoder_output_dim = encoder_output_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_output_dim = decoder_output_dim
        self.max_target_seq_len = max_target_seq_len
        self.output_common_embedding = output_common_embedding
        self.output_common_vocab = output_common_vocab
        assert (output_common_embedding is None) ^ (output_common_vocab is not None)
        assert output_common_embedding is None or \
               output_common_embedding.weight.size(1) == self.decoder_output_dim
        self.nr_output_common_embeddings = 0 if self.output_common_embedding is None else \
            self.output_common_embedding.weight.size(0)
        assert output_common_vocab is None or len(output_common_vocab) == self.nr_output_common_embeddings
        self.nr_rnn_layers = nr_rnn_layers
        rnn_type = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.decoding_rnn_layer = rnn_type(
            input_size=self.decoder_hidden_dim, hidden_size=self.decoder_hidden_dim, num_layers=self.nr_rnn_layers)

        self.attn_and_input_combine_linear_layer = nn.Linear(
            self.encoder_output_dim + self.decoder_output_dim, self.decoder_hidden_dim)
        self.output_common_embedding_dropout_layer = None if embedding_dropout_rate is None else nn.Dropout(embedding_dropout_rate)
        self.out_linear_layer = nn.Linear(self.decoder_hidden_dim, self.decoder_output_dim)
        self.attention_over_encoder_outputs = Attention(
            in_embed_dim=self.encoder_output_dim, project_key=True, project_query=True,
            query_in_embed_dim=self.decoder_hidden_dim + self.decoder_output_dim, activation_fn=activation_fn)
        self.dyn_vocab_linear_projection = nn.Linear(1028, 256)  # TODO: plug-in HPs
        self.dyn_vocab_strategy = 'after_softmax'  # in {'before_softmax', 'after_softmax'}

    def forward(self, encoder_outputs: torch.Tensor,
                encoder_outputs_mask: Optional[torch.BoolTensor] = None,
                output_vocab_batch_flattened_encodings: Optional[torch.Tensor] = None,
                output_vocab_example_based_encodings: Optional[torch.Tensor] = None,
                output_vocab_example_based_encodings_mask: Optional[torch.BoolTensor] = None,
                dyn_vocab_scattered_encodings: Optional[ScatteredEncodings] = None,
                groundtruth_target_idxs: Optional[torch.LongTensor] = None):
        assert encoder_outputs.ndim == 3  # (batch_size, encoder_output_len, encoder_output_dim)
        batch_size, encoder_output_len = encoder_outputs.size()[:2]
        assert self.encoder_output_dim == encoder_outputs.size(2)
        assert output_vocab_example_based_encodings is None or \
               output_vocab_example_based_encodings.ndim == 3 and \
               output_vocab_example_based_encodings.size(0) == batch_size and \
               output_vocab_example_based_encodings.size(2) == self.decoder_output_dim
        assert output_vocab_batch_flattened_encodings is None or output_vocab_example_based_encodings is None  # cannot be set together

        if output_vocab_example_based_encodings is not None and output_vocab_batch_flattened_encodings is None:
            nr_output_batched_words_per_example = output_vocab_example_based_encodings.size(1)
        elif output_vocab_batch_flattened_encodings is not None and output_vocab_example_based_encodings is None:
            nr_output_batched_words_per_example = output_vocab_batch_flattened_encodings.size(0)
        else:
            raise ValueError(
                'Must specify exactly one of `output_batched_flattened_encodings` and `output_batched_encodings`.')

        nr_all_possible_output_words_encodings = nr_output_batched_words_per_example + self.nr_output_common_embeddings
        assert output_vocab_example_based_encodings_mask is None or output_vocab_example_based_encodings_mask.size() == \
               (batch_size, nr_output_batched_words_per_example)
        assert encoder_outputs_mask is None or encoder_outputs_mask.size() == (batch_size, encoder_output_len)
        # encoder_outputs_mask = None if encoder_outputs_mask is None else encoder_outputs_mask[:, :encoder_output_len]

        # TODO: export to `init_hidden()` method.
        hidden_shape = (self.nr_rnn_layers, batch_size, self.decoder_hidden_dim)
        rnn_hidden = torch.zeros(hidden_shape, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
        rnn_state = torch.zeros(hidden_shape, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
        outputs = []

        assert (groundtruth_target_idxs is None) ^ self.training  # used only while training with teacher-enforcing

        target_encodings = None
        prev_cell_output_idx = None
        target_seq_len = self.max_target_seq_len
        if self.training:
            assert groundtruth_target_idxs is not None
            assert groundtruth_target_idxs.ndim == 2 and groundtruth_target_idxs.size(0) == batch_size
            target_seq_len = groundtruth_target_idxs.size(1)
            assert target_seq_len <= self.max_target_seq_len
            target_encodings = apply_embeddings(
                indices=groundtruth_target_idxs,
                common_embeddings=self.output_common_embedding,
                batched_flattened_encodings=output_vocab_batch_flattened_encodings,
                batched_encodings=output_vocab_example_based_encodings)
            assert target_encodings.size() == (batch_size, target_seq_len, self.decoder_output_dim)
            if self.output_common_embedding_dropout_layer is not None:
                target_encodings = self.output_common_embedding_dropout_layer(
                    target_encodings)  # (batch_size, target_seq_len, decoder_output_dim)  # TODO: insert the dropout application into `apply_batched_embeddings()`
        else:
            prev_cell_output_idx = torch.tensor(
                [self.output_common_vocab.get_word_idx('<SOS>')],
                device=encoder_outputs.device, dtype=torch.long).expand(batch_size,)

        for T in range(target_seq_len - 1):  # we don't have to predict the initial `<SOS>` special word.
            if self.training:
                prev_cell_encoding = target_encodings[:, T, :]
            else:
                prev_cell_encoding = apply_embeddings(
                    indices=prev_cell_output_idx,
                    common_embeddings=self.output_common_embedding,
                    batched_flattened_encodings=output_vocab_batch_flattened_encodings,
                    batched_encodings=output_vocab_example_based_encodings)
            assert prev_cell_encoding.size() == (batch_size, self.decoder_output_dim)

            attn_query_from = torch.cat((prev_cell_encoding, rnn_hidden[-1, :, :]), dim=1)
            attn_applied = self.attention_over_encoder_outputs(
                sequences=encoder_outputs, query=attn_query_from, mask=encoder_outputs_mask)
            assert attn_applied.size() == (batch_size, self.encoder_output_dim)

            attn_applied_and_input_combine = torch.cat((prev_cell_encoding, attn_applied), dim=1)  # (batch_size, decoder_output_dim + encoder_output_dim)
            attn_applied_and_input_combine = self.attn_and_input_combine_linear_layer(attn_applied_and_input_combine)  # (batch_size, decoder_hidden_dim)

            rnn_cell_input = self.activation_layer(attn_applied_and_input_combine).unsqueeze(0)  # (1, batch_size, decoder_hidden_dim)
            rnn_cell_output, (rnn_hidden, rnn_state) = self.decoding_rnn_layer(rnn_cell_input, (rnn_hidden, rnn_state))
            assert rnn_cell_output.size() == (1, batch_size, self.decoder_hidden_dim)
            assert rnn_hidden.size() == hidden_shape and rnn_state.size() == hidden_shape

            next_output_after_linear = self.out_linear_layer(rnn_cell_output[0])  # (batch_size, decoder_output_dim)

            if output_vocab_example_based_encodings is not None:
                projection_on_batched_target_encodings_wo_common = torch.bmm(
                    output_vocab_example_based_encodings, next_output_after_linear.unsqueeze(-1))\
                    .view(batch_size, nr_output_batched_words_per_example)
                if output_vocab_example_based_encodings_mask is not None:
                    # TODO: does it really makes sense to mask this?
                    projection_on_batched_target_encodings_wo_common = projection_on_batched_target_encodings_wo_common + torch.where(
                        output_vocab_example_based_encodings_mask,
                        torch.zeros(1, dtype=torch.float, device=projection_on_batched_target_encodings_wo_common.device),
                        torch.full(size=(1,), fill_value=float('-inf'), dtype=torch.float,
                                   device=projection_on_batched_target_encodings_wo_common.device))
            elif output_vocab_batch_flattened_encodings is not None:
                # (batch_size, decoder_output_dim) * (nr_symbols, decoder_output_dim).T
                #   -> (batch_size, nr_symbols)
                projection_on_batched_target_encodings_wo_common = torch.mm(
                    next_output_after_linear, output_vocab_batch_flattened_encodings.permute(1, 0)) \
                    .view(batch_size, nr_output_batched_words_per_example)
            assert projection_on_batched_target_encodings_wo_common.size() == \
                   (batch_size, nr_output_batched_words_per_example)

            if dyn_vocab_scattered_encodings is not None:
                dyn_vocab_scattered_encodings_projected = self.dyn_vocab_linear_projection(
                    dyn_vocab_scattered_encodings.encodings)
                projection_on_dyn_vocab_scattered_encodings_wo_common = torch.bmm(
                    dyn_vocab_scattered_encodings_projected, next_output_after_linear.unsqueeze(-1)) \
                    .view(batch_size, dyn_vocab_scattered_encodings.encodings.size(1))

                if self.dyn_vocab_strategy == 'after_softmax':
                    # Use this masking is summing occurrences AFTER applying softmax
                    projection_on_dyn_vocab_scattered_encodings_wo_common = \
                        projection_on_dyn_vocab_scattered_encodings_wo_common + torch.where(
                            dyn_vocab_scattered_encodings.mask,  # (bsz, max_nr_symbols_occurrences)
                            torch.zeros(1, dtype=torch.float,
                                        device=projection_on_dyn_vocab_scattered_encodings_wo_common.device),
                            torch.full(size=(1,), fill_value=float('-inf'), dtype=torch.float,
                                       device=projection_on_dyn_vocab_scattered_encodings_wo_common.device))
                else:
                    # Use this masking if summing occurrences BEFORE applying softmax
                    projection_on_dyn_vocab_scattered_encodings_wo_common = \
                        projection_on_dyn_vocab_scattered_encodings_wo_common.masked_fill(
                            ~dyn_vocab_scattered_encodings.mask, 0)  # (bsz, max_nr_symbols_occurrences)

            # (batch_size, decoder_output_dim) * (nr_output_common_embeddings, decoder_output_dim).T
            #   -> (batch_size, nr_output_common_embeddings)
            projection_on_output_common_embeddings = torch.mm(
                next_output_after_linear, self.output_common_embedding.weight.t())
            assert projection_on_output_common_embeddings.size() == (batch_size, self.nr_output_common_embeddings)

            if dyn_vocab_scattered_encodings is None:
                projection_on_all_output_encodings = torch.cat(
                    (projection_on_output_common_embeddings, projection_on_batched_target_encodings_wo_common),
                    dim=-1)
            else:
                if self.dyn_vocab_strategy == 'after_softmax':
                    # Use this masking is summing occurrences AFTER applying softmax
                    projection_on_all_output_encodings = torch.cat(
                        (projection_on_output_common_embeddings,
                         projection_on_batched_target_encodings_wo_common,
                         projection_on_dyn_vocab_scattered_encodings_wo_common),
                        dim=-1)
                else:
                    # Use this masking is summing occurrences BEFORE applying softmax
                    projection_on_all_output_encodings = torch.cat(
                        (projection_on_output_common_embeddings, projection_on_batched_target_encodings_wo_common), dim=-1)
                    projection_on_all_output_encodings = projection_on_all_output_encodings.scatter_add(
                        dim=1,
                        index=dyn_vocab_scattered_encodings.indices + self.nr_output_common_embeddings,  # We assume the indexing here is w/o the common
                        src=projection_on_dyn_vocab_scattered_encodings_wo_common)

            if dyn_vocab_scattered_encodings is None:
                final_cell_log_softmax_out = F.log_softmax(projection_on_all_output_encodings, dim=1)
            else:
                if self.dyn_vocab_strategy == 'after_softmax':
                    # Use this masking is summing occurrences AFTER applying softmax
                    rnn_cell_output = F.softmax(projection_on_all_output_encodings, dim=1)
                    rnn_cell_output_no_dyn_vocab = rnn_cell_output[:, :nr_all_possible_output_words_encodings]
                    assert rnn_cell_output_no_dyn_vocab.size() == (batch_size, nr_all_possible_output_words_encodings)
                    rnn_cell_output_dyn_vocab = rnn_cell_output[:, nr_all_possible_output_words_encodings:]
                    final_cell_softmax_out = rnn_cell_output_no_dyn_vocab.scatter_add(
                        dim=1,
                        index=dyn_vocab_scattered_encodings.indices + self.nr_output_common_embeddings,  # We assume the indexing here is w/o the common
                        src=rnn_cell_output_dyn_vocab)
                    final_cell_softmax_out = final_cell_softmax_out + torch.finfo().eps
                    final_cell_log_softmax_out = final_cell_softmax_out.log()
                else:
                    # Use this masking is summing occurrences BEFORE applying softmax
                    final_cell_log_softmax_out = F.log_softmax(projection_on_all_output_encodings, dim=1)
            outputs.append(final_cell_log_softmax_out)

            if not self.training:
                _, prev_cell_output_idx = final_cell_log_softmax_out.topk(1, dim=1)
                prev_cell_output_idx = prev_cell_output_idx.squeeze(dim=1)
                assert prev_cell_output_idx.size() == (batch_size,)
                assert prev_cell_output_idx.dtype == torch.long

        outputs = torch.stack(outputs).permute(1, 0, 2)
        assert outputs.size() == (batch_size, target_seq_len - 1, nr_all_possible_output_words_encodings)  # w/o <SOS>
        return outputs

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from ddfa.code_nn_modules.vocabulary import Vocabulary
from ddfa.nn_utils.apply_batched_embeddings import apply_batched_embeddings
from ddfa.nn_utils.attention import Attention


__all__ = ['AttnRNNDecoder']


class AttnRNNDecoder(nn.Module):
    def __init__(self, encoder_output_len: int, encoder_output_dim: int, decoder_hidden_dim: int,
                 decoder_output_dim: int, max_target_seq_len: int,  embedding_dropout_p: Optional[float] = 0.3,
                 rnn_type: str = 'lstm', nr_rnn_layers: int = 1,
                 output_common_embedding: Optional[Union[torch.Tensor, nn.Embedding]] = None,
                 output_common_vocab: Optional[Vocabulary] = None):
        assert rnn_type in {'lstm', 'gru'}
        super(AttnRNNDecoder, self).__init__()
        self.encoder_output_len = encoder_output_len # TODO: remove!
        self.encoder_output_dim = encoder_output_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_output_dim = decoder_output_dim
        self.max_target_seq_len = max_target_seq_len
        self.output_common_embedding = output_common_embedding
        self.output_common_vocab = output_common_vocab
        assert (output_common_embedding is None) ^ (output_common_vocab is not None)
        assert output_common_embedding is None or \
               output_common_embedding.weight.size()[1] == self.decoder_output_dim
        self.nr_output_common_embeddings = 0 if self.output_common_embedding is None else \
            self.output_common_embedding.weight.size()[0]
        assert output_common_vocab is None or len(output_common_vocab) == self.nr_output_common_embeddings
        self.nr_rnn_layers = nr_rnn_layers
        rnn_type = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.decoding_rnn_layer = rnn_type(
            input_size=self.decoder_hidden_dim, hidden_size=self.decoder_hidden_dim, num_layers=self.nr_rnn_layers)

        self.attn_and_input_combine_linear_layer = nn.Linear(
            self.decoder_hidden_dim + self.decoder_output_dim, self.decoder_hidden_dim)
        self.output_common_embedding_dropout_layer = None if embedding_dropout_p is None else nn.Dropout(embedding_dropout_p)
        self.out_linear_layer = nn.Linear(self.decoder_hidden_dim, self.decoder_output_dim)
        self.attention_over_encoder_outputs = Attention(
            nr_features=self.encoder_output_dim, project_key=True, project_query=True,
            key_in_features=self.decoder_hidden_dim + self.decoder_output_dim)

    def forward(self, encoder_outputs: torch.Tensor,
                encoder_outputs_mask: Optional[torch.BoolTensor] = None,
                output_batched_encodings: Optional[torch.Tensor] = None,
                output_batched_encodings_mask: Optional[torch.BoolTensor] = None,
                target_idxs: Optional[torch.LongTensor] = None):
        assert len(encoder_outputs.size()) == 3  # (batch_size, encoder_output_len, encoder_output_dim)
        batch_size, encoder_output_len, encoder_output_dim = encoder_outputs.size()
        assert output_batched_encodings is None or \
               len(output_batched_encodings.size()) == 3 and \
               output_batched_encodings.size()[0] == batch_size and \
               output_batched_encodings.size()[2] == self.decoder_output_dim
        nr_output_batched_words_per_example = output_batched_encodings.size()[1]
        nr_all_possible_output_words_encodings = nr_output_batched_words_per_example + self.nr_output_common_embeddings
        assert output_batched_encodings_mask is None or output_batched_encodings_mask.size() == \
               (batch_size, nr_output_batched_words_per_example)
        assert encoder_output_len <= self.encoder_output_len
        assert encoder_output_dim == self.encoder_output_dim
        assert encoder_outputs_mask is None or encoder_outputs_mask.size() == (batch_size, self.encoder_output_len)
        encoder_outputs_mask = None if encoder_outputs_mask is None else encoder_outputs_mask[:, :encoder_output_len]

        # TODO: export to `init_hidden()` method.
        hidden_shape = (self.nr_rnn_layers, batch_size, self.decoder_hidden_dim)
        rnn_hidden = torch.zeros(hidden_shape, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
        rnn_state = torch.zeros(hidden_shape, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
        outputs = []

        assert (target_idxs is None) ^ self.training  # used only while training with teacher-enforcing

        target_encodings = None
        prev_cell_output_idx = None
        target_seq_len = self.max_target_seq_len
        if self.training:
            assert target_idxs is not None
            assert len(target_idxs.size()) == 2 and target_idxs.size()[0] == batch_size
            target_seq_len = target_idxs.size()[1]
            assert target_seq_len <= self.max_target_seq_len
            target_encodings = apply_batched_embeddings(
                batched_embeddings=output_batched_encodings, indices=target_idxs,
                common_embeddings=self.output_common_embedding)  # (batch_size, target_seq_len, decoder_output_dim)
            assert target_encodings.size() == (batch_size, target_seq_len, self.decoder_output_dim)
            if self.output_common_embedding_dropout_layer is not None:
                target_encodings = self.output_common_embedding_dropout_layer(
                    target_encodings)  # (batch_size, target_seq_len, decoder_output_dim)  # TODO: insert the dropout application into `apply_batched_embeddings()`
        else:
            prev_cell_output_idx = torch.tensor(
                [self.output_common_vocab.get_word_idx_or_unk('<SOS>')],
                device=encoder_outputs.device, dtype=torch.long).expand(batch_size,)

        for T in range(target_seq_len - 1):  # we don't have to predict the initial `<SOS>` special word.
            if self.training:
                prev_cell_encoding = target_encodings[:, T, :]
            else:
                prev_cell_encoding = apply_batched_embeddings(
                    batched_embeddings=output_batched_encodings, indices=prev_cell_output_idx,
                    common_embeddings=self.output_common_embedding)
            assert prev_cell_encoding.size() == (batch_size, self.decoder_output_dim)

            attn_key_from = torch.cat((prev_cell_encoding, rnn_hidden[-1, :, :]), dim=1)
            attn_applied = self.attention_over_encoder_outputs(
                sequences=encoder_outputs, attn_key_from=attn_key_from, mask=encoder_outputs_mask)
            assert attn_applied.size() == (batch_size, self.encoder_output_dim)

            attn_applied_and_input_combine = torch.cat((prev_cell_encoding, attn_applied), dim=1)  # (batch_size, decoder_output_dim + encoder_output_dim)
            attn_applied_and_input_combine = self.attn_and_input_combine_linear_layer(attn_applied_and_input_combine)  # (batch_size, decoder_hidden_dim)

            rnn_cell_input = F.relu(attn_applied_and_input_combine).unsqueeze(0)  # (1, batch_size, decoder_hidden_dim)
            rnn_cell_output, (rnn_hidden, rnn_state) = self.decoding_rnn_layer(rnn_cell_input, (rnn_hidden, rnn_state))
            assert rnn_cell_output.size() == (1, batch_size, self.decoder_hidden_dim)
            assert rnn_hidden.size() == hidden_shape and rnn_state.size() == hidden_shape

            next_output_after_linear = self.out_linear_layer(rnn_cell_output[0])  # (batch_size, decoder_output_dim)

            projection_on_batched_target_encodings_wo_common = torch.bmm(
                output_batched_encodings, next_output_after_linear.unsqueeze(-1))\
                .view(batch_size, nr_output_batched_words_per_example)
            if output_batched_encodings_mask is not None:
                # TODO: does it really makes sense to mask this?
                projection_on_batched_target_encodings_wo_common += torch.where(
                    output_batched_encodings_mask,
                    torch.zeros(1, dtype=torch.float, device=projection_on_batched_target_encodings_wo_common.device),
                    torch.full(size=(1,), fill_value=float('-inf'), dtype=torch.float,
                               device=projection_on_batched_target_encodings_wo_common.device))
            assert projection_on_batched_target_encodings_wo_common.size() == (batch_size, nr_output_batched_words_per_example)

            # (batch_size, decoder_output_dim) * (nr_output_common_embeddings, decoder_output_dim).T
            #   -> (batch_size, nr_output_common_embeddings)
            projection_on_output_common_embeddings = torch.mm(
                next_output_after_linear, self.output_common_embedding.weight.t())
            assert projection_on_output_common_embeddings.size() == (batch_size, self.nr_output_common_embeddings)

            projection_on_all_output_encodings = torch.cat(
                (projection_on_output_common_embeddings, projection_on_batched_target_encodings_wo_common), dim=-1)

            rnn_cell_output = F.log_softmax(projection_on_all_output_encodings, dim=1)
            assert rnn_cell_output.size() == (batch_size, nr_all_possible_output_words_encodings)
            outputs.append(rnn_cell_output)

            if not self.training:
                _, prev_cell_output_idx = rnn_cell_output.topk(1, dim=1)
                prev_cell_output_idx = prev_cell_output_idx.squeeze(dim=1)
                assert prev_cell_output_idx.size() == (batch_size,) and prev_cell_output_idx.dtype == torch.long

        outputs = torch.stack(outputs).permute(1, 0, 2)
        assert outputs.size() == (batch_size, target_seq_len - 1, nr_all_possible_output_words_encodings)  # w/o <SOS>
        return outputs

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing

from ddfa.code_nn_modules.vocabulary import Vocabulary
from ddfa.nn_utils.apply_batched_embeddings import apply_batched_embeddings


class SymbolsDecoder(nn.Module):
    def __init__(self, symbols_special_words_vocab: Vocabulary, symbols_special_words_embedding: nn.Embedding,
                 input_len: int = 80, input_dim: int = 256, symbols_encoding_dim: int = 256,
                 symbols_emb_dropout_p: float = 0.1):
        super(SymbolsDecoder, self).__init__()
        self.encoder_output_len = input_len
        self.encoder_output_dim = input_dim
        self.symbols_encoding_dim = symbols_encoding_dim
        self.symbols_special_words_vocab = symbols_special_words_vocab
        self.symbols_special_words_embedding = symbols_special_words_embedding
        self.nr_rnn_layers = 2
        self.decoding_lstm_layer = nn.LSTM(
            input_size=self.symbols_encoding_dim, hidden_size=self.symbols_encoding_dim, num_layers=self.nr_rnn_layers)

        self.attn_weights_linear_layer = nn.Linear(self.symbols_encoding_dim * 2, self.encoder_output_len)
        self.attn_and_input_combine_linear_layer = nn.Linear(
            self.symbols_encoding_dim + self.encoder_output_dim, self.symbols_encoding_dim)
        self.symbols_emb_dropout_p = symbols_emb_dropout_p
        self.symbols_emb_dropout_layer = nn.Dropout(self.symbols_emb_dropout_p)
        self.out_linear_layer = nn.Linear(self.symbols_encoding_dim, self.symbols_encoding_dim)

    def forward(self, encoder_outputs: torch.Tensor, encoder_outputs_mask: typing.Optional[torch.BoolTensor],
                symbols_encodings: torch.Tensor, symbols_encodings_mask: typing.Optional[torch.BoolTensor],
                target_symbols_idxs: typing.Optional[torch.LongTensor]):
        assert len(encoder_outputs.size()) == 3  # (batch_size, encoder_output_len, encoder_output_dim)
        batch_size, encoder_output_len, encoder_output_dim = encoder_outputs.size()
        assert len(symbols_encodings.size()) == 3
        assert symbols_encodings.size()[0] == batch_size and symbols_encodings.size()[2] == self.symbols_encoding_dim
        nr_all_symbols_wo_specials = symbols_encodings.size()[1]
        nr_all_symbols_with_specials = nr_all_symbols_wo_specials + len(self.symbols_special_words_vocab)
        assert symbols_encodings_mask is None or symbols_encodings_mask.size() == (batch_size, nr_all_symbols_wo_specials)
        assert encoder_output_len == self.encoder_output_len
        assert encoder_output_dim == self.encoder_output_dim
        assert encoder_outputs_mask is None or encoder_outputs_mask.size() == (batch_size, encoder_output_len)

        assert (target_symbols_idxs is None) ^ self.training  # used only while training with teacher-enforcing
        if not self.training:
            raise NotImplementedError  # TODO: implement

        assert target_symbols_idxs is not None
        assert len(target_symbols_idxs.size()) == 2 and target_symbols_idxs.size()[0] == batch_size
        nr_target_symbols = target_symbols_idxs.size()[1]
        target_symbols_encodings = apply_batched_embeddings(
            batched_embeddings=symbols_encodings, indices=target_symbols_idxs,
            common_embeddings=self.symbols_special_words_embedding)  # (batch_size, nr_target_symbols, symbols_encoding_dim)
        assert target_symbols_encodings.size() == (batch_size, nr_target_symbols, self.symbols_encoding_dim)
        target_symbols_encodings = self.symbols_emb_dropout_layer(
            target_symbols_encodings)  # (batch_size, nr_target_symbols, symbols_encoding_dim)

        hidden_shape = (self.nr_rnn_layers, batch_size, self.symbols_encoding_dim)
        rnn_hidden = torch.zeros(hidden_shape, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
        rnn_state = torch.zeros(hidden_shape, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
        outputs = []
        for T in range(nr_target_symbols - 1):  # we don't have to predict the initial `<SOS>` special word.
            attn_weights = self.attn_weights_linear_layer(
                torch.cat((target_symbols_encodings[:, T, :], rnn_hidden[-1, :, :]), dim=1))  # (batch_size, encoder_output_len)
            assert attn_weights.size() == (batch_size, encoder_output_len)
            if encoder_outputs_mask is not None:
                # attn_weights.masked_fill_(~encoder_outputs_mask, float('-inf'))
                attn_weights = attn_weights + torch.where(
                    encoder_outputs_mask, torch.zeros_like(attn_weights), torch.full_like(attn_weights, float('-inf')))
            attn_probs = F.softmax(attn_weights, dim=1)  # (batch_size, encoder_output_len)
            # (batch_size, 1, encoder_output_len) * (batch_size, encoder_output_len, encoder_output_dim)
            # = (batch_size, 1, encoder_output_dim)
            attn_applied = torch.bmm(
                attn_probs.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, encoder_output_dim)

            attn_combine = torch.cat((target_symbols_encodings[:, T, :], attn_applied), dim=1)  # (batch_size, symbols_encoding_dim + encoder_output_dim)
            attn_combine = self.attn_and_input_combine_linear_layer(attn_combine)  # (batch_size, symbols_encoding_dim)

            rnn_cell_input = F.relu(attn_combine).unsqueeze(0)  # (1, batch_size, symbols_encoding_dim)
            output, (rnn_hidden, rnn_state) = self.decoding_lstm_layer(rnn_cell_input, (rnn_hidden, rnn_state))
            assert output.size() == (1, batch_size, self.symbols_encoding_dim)
            assert rnn_hidden.size() == hidden_shape and rnn_state.size() == hidden_shape

            next_symbol_output_after_linear = self.out_linear_layer(output[0])  # (batch_size, self.symbols_encoding_dim)

            projection_on_symbols_encodings_wo_specials = torch.bmm(
                symbols_encodings, next_symbol_output_after_linear.unsqueeze(-1))\
                .view(batch_size, nr_all_symbols_wo_specials)
            if symbols_encodings_mask is not None:
                # TODO: does it really makes sense to mask this?
                projection_on_symbols_encodings_wo_specials += torch.where(
                    symbols_encodings_mask,
                    torch.zeros_like(projection_on_symbols_encodings_wo_specials),
                    torch.full_like(projection_on_symbols_encodings_wo_specials, float('-inf')))
            assert projection_on_symbols_encodings_wo_specials.size() == (batch_size, nr_all_symbols_wo_specials)

            assert self.symbols_special_words_embedding.weight.size() == (len(self.symbols_special_words_vocab), self.symbols_encoding_dim)
            # (batch_size, symbols_encoding_dim) * (nr_specials, symbols_encoding_dim).T = (batch_size, nr_specials)
            projection_on_symbols_special_words_encodings = torch.mm(
                next_symbol_output_after_linear, self.symbols_special_words_embedding.weight.t())
            assert projection_on_symbols_special_words_encodings.size() == (batch_size, len(self.symbols_special_words_vocab))

            projection_on_symbols_encodings = torch.cat(
                (projection_on_symbols_special_words_encodings, projection_on_symbols_encodings_wo_specials), dim=-1)

            output = F.log_softmax(projection_on_symbols_encodings, dim=1)
            assert output.size() == (batch_size, nr_all_symbols_with_specials)
            outputs.append(output)

        outputs = torch.stack(outputs).permute(1, 0, 2)
        assert outputs.size() == (batch_size, nr_target_symbols - 1, nr_all_symbols_with_specials)
        return outputs

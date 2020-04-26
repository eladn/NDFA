import torch
import torch.nn as nn
import torch.nn.functional as F
import typing

from ddfa.code_nn_modules.vocabulary import Vocabulary
from ddfa.nn_utils import apply_batched_embeddings


class SymbolsDecoder(nn.Module):
    def __init__(self, symbols_special_words_vocab: Vocabulary, symbols_special_words_embedding: nn.Embedding,
                 input_len: int = 80, input_dim: int = 256, symbols_encoding_dim: int = 256, dropout_p: float = 0.1):
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
        self.symbols_emb_dropout_layer = nn.Dropout(self.dropout_p)
        self.out_linear_layer = nn.Linear(self.encoder_output_dim, self.symbols_encoding_dim)

    def forward(self, encoder_outputs: torch.Tensor, encoder_outputs_mask: typing.Optional[torch.Tensor],
                symbols_encodings: torch.Tensor, target_symbols_idxs: typing.Optional[torch.IntTensor]):
        assert len(encoder_outputs.size()) == 3  # (batch_size, encoder_output_len, encoder_output_dim)
        batch_size, encoder_output_len, encoder_output_dim = encoder_outputs.size()
        assert encoder_output_len == self.encoder_output_len
        assert encoder_output_dim == self.encoder_output_dim

        assert (target_symbols_idxs is None) ^ self.training  # used only while training with teacher-enforcing
        if not self.training:
            raise NotImplementedError  # TODO: implement

        assert target_symbols_idxs is not None
        assert len(target_symbols_idxs.size()) == 2 and target_symbols_idxs.size()[0] == batch_size
        nr_symbols = target_symbols_idxs.size()[1]
        target_symbols_encodings = apply_batched_embeddings(
            batched_embeddings=symbols_encodings, indices=target_symbols_idxs,
            common_embeddings=self.symbols_special_words_embedding)  # (batch_size, nr_symbols, symbols_encoding_dim)
        assert target_symbols_encodings.size() == (batch_size, nr_symbols, self.symbols_encoding_dim)
        target_symbols_encodings = self.symbols_emb_dropout_layer(
            target_symbols_encodings)  # (batch_size, nr_symbols, symbols_encoding_dim)

        hidden_shape = (self.nr_rnn_layers, batch_size, self.symbols_encoding_dim)
        rnn_hidden = torch.zeros(hidden_shape, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
        rnn_state = torch.zeros(hidden_shape, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
        outputs = []
        for T in range(nr_symbols):
            attn_weights = self.attn_weights_linear_layer(
                torch.cat((target_symbols_encodings[:, T, :], rnn_hidden[-1, :, :]), dim=1))  # (batch_size, encoder_output_len)
            assert attn_weights.size() == (batch_size, encoder_output_len)
            attn_probs = F.softmax(attn_weights, dim=1)  # (batch_size, encoder_output_len)
            # (batch_size, 1, encoder_output_len) * (batch_size, encoder_output_len, encoder_output_dim)
            # = (batch_size, 1, encoder_output_dim)
            attn_applied = torch.bmm(
                attn_probs.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, encoder_output_dim)

            attn_combine = torch.cat((target_symbols_encodings[:, T, :], attn_applied), dim=1)  # (batch_size, symbols_encoding_dim + encoder_output_dim)
            attn_combine = self.attn_and_input_combine_linear_layer(attn_combine)  # (batch_size, symbols_encoding_dim)

            rnn_cell_input = F.relu(attn_combine).unsqueeze(0)  # (1, batch_size, symbols_encoding_dim)
            output, (rnn_hidden, rnn_state) = self.decoding_lstm_layer(rnn_cell_input, rnn_hidden, rnn_state)
            assert output.size() == (1, batch_size, self.symbols_encoding_dim)
            assert rnn_hidden.size() == hidden_shape and rnn_state.size() == hidden_shape

            output = F.log_softmax(self.out(output[0]), dim=1)
            outputs.append(output)

        return outputs

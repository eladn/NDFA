import torch
import torch.nn as nn
import typing

from ddfa.nn_utils.attn_rnn_decoder import AttnRNNDecoder


class SymbolsDecoder(nn.Module):
    def __init__(self, symbols_special_words_embedding: nn.Embedding,
                 encoder_output_len: int = 80, encoder_output_dim: int = 256, symbols_encoding_dim: int = 256,
                 symbols_emb_dropout_p: float = 0.1):
        super(SymbolsDecoder, self).__init__()
        self.attn_rnn_decoder = AttnRNNDecoder(
            encoder_output_len=encoder_output_len, encoder_output_dim=encoder_output_dim,
            decoder_hidden_dim=max(symbols_encoding_dim * 4, encoder_output_dim),
            decoder_output_dim=symbols_encoding_dim, embedding_dropout_p=symbols_emb_dropout_p,
            rnn_type='lstm', nr_rnn_layers=2, output_common_embedding=symbols_special_words_embedding)

    def forward(self, encoder_outputs: torch.Tensor, encoder_outputs_mask: typing.Optional[torch.BoolTensor],
                symbols_encodings: torch.Tensor, symbols_encodings_mask: typing.Optional[torch.BoolTensor],
                target_symbols_idxs: typing.Optional[torch.LongTensor]):
        return self.attn_rnn_decoder(
            encoder_outputs=encoder_outputs, encoder_outputs_mask=encoder_outputs_mask,
            output_batched_encodings=symbols_encodings, output_batched_encodings_mask=symbols_encodings_mask,
            target_idxs=target_symbols_idxs)

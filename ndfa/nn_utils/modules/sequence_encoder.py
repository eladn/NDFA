import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm
import dataclasses
from typing import Optional

from ndfa.ndfa_model_hyper_parameters import SequenceEncoderParams
from ndfa.nn_utils.modules.rnn_encoder import RNNEncoder
from ndfa.nn_utils.modules.sequence_combiner import SequenceCombiner


__all__ = ['EncodedSequence', 'SequenceEncoder']


@dataclasses.dataclass
class EncodedSequence:
    sequence: torch.Tensor
    last_seq_element: Optional[torch.Tensor] = None
    combined: Optional[torch.Tensor] = None


class SequenceEncoder(nn.Module):
    def __init__(self, encoder_params: SequenceEncoderParams,
                 input_dim: int, hidden_dim: Optional[int] = None,
                 combined_dim: Optional[int] = None,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(SequenceEncoder, self).__init__()
        self.encoder_params = encoder_params
        if self.encoder_params.encoder_type == 'rnn':
            # self.attn_rnn_encoder = AttnRNNEncoder(
            #     input_dim=input_dim, hidden_dim=hidden_dim,
            #     rnn_type=self.encoder_params.rnn_type,
            #     nr_rnn_layers=self.encoder_params.nr_rnn_layers,
            #     rnn_bi_direction=self.encoder_params.rnn_bi_direction,
            #     activation_fn=activation_fn)
            self.rnn_encoder = RNNEncoder(
                input_dim=input_dim, hidden_dim=hidden_dim,
                rnn_type=self.encoder_params.rnn_type,
                nr_rnn_layers=self.encoder_params.nr_rnn_layers,
                rnn_bi_direction=self.encoder_params.bidirectional_rnn)
        elif self.encoder_params.encoder_type == 'transformer':
            raise NotImplementedError
            # FROM EXPRESSION ENCODER:
            # transformer_encoder_layer = TransformerEncoderLayer(
            #     d_model=self.encoder_params.token_encoding_dim, nhead=1)
            # encoder_norm = LayerNorm(self.encoder_params.token_encoding_dim)
            # self.transformer_encoder = TransformerEncoder(
            #     encoder_layer=transformer_encoder_layer, num_layers=3, norm=encoder_norm)
        else:
            raise ValueError(f'Unsupported sequence encoder type {self.encoder_params.encoder_type}.')

        if self.encoder_params.sequence_combiner is not None:
            self.sequence_combiner = SequenceCombiner(
                encoding_dim=hidden_dim, combined_dim=combined_dim,
                combiner_params=self.encoder_params.sequence_combiner,
                dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(self, sequence_input: torch.Tensor, mask: Optional[torch.BoolTensor] = None,
                lengths: Optional[torch.LongTensor] = None, batch_first: bool = True,
                sorted_by_length: bool = False) -> EncodedSequence:
        if not batch_first:
            raise NotImplementedError
        sequence_output, last_seq_element_output, combined_sequence_outputs = None, None, None

        if self.encoder_params.encoder_type == 'rnn':
            # We do not use `attn_rnn_encoder` anymore. Now we use `sequence_combiner` instead.
            # combined_sequence_outputs, sequence_output = self.attn_rnn_encoder(
            #     sequence_input=sequence_input, mask=mask, lengths=lengths, batch_first=batch_first)
            last_seq_element_output, sequence_output = self.rnn_encoder(
                sequence_input=sequence_input, mask=mask, lengths=lengths,
                batch_first=batch_first, sorted_by_length=sorted_by_length)
        elif self.encoder_params.encoder_type == 'transformer':
            raise NotImplementedError
            # FROM EXPRESSION ENCODER:
            # expr_embeddings_projected_SNE = expr_embeddings_projected.permute(1, 0, 2)  # (nr_tokens_in_expr, bsz, embedding_dim)
            # if expressions_mask is not None:
            #     expressions_mask = ~expressions_mask.flatten(0, 1)  # (bsz * nr_exprs, nr_tokens_in_expr)
            # full_expr_encoded = self.transformer_encoder(
            #     expr_embeddings_projected_SNE,
            #     src_key_padding_mask=expressions_mask)
            # expr_encoded_merge = full_expr_encoded.sum(dim=0).view(batch_size, nr_exprs, -1)
        else:
            raise ValueError(f'Unsupported sequence encoder type {self.encoder_params.encoder_type}.')

        if self.encoder_params.sequence_combiner is not None:
            combined_sequence_outputs = self.sequence_combiner(
                sequence_encodings=sequence_output,
                sequence_mask=mask,
                sequence_lengths=lengths,
                batch_first=batch_first)

        return EncodedSequence(
            sequence=sequence_output,
            last_seq_element=last_seq_element_output,
            combined=combined_sequence_outputs)

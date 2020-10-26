import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm
from typing import Optional

from ndfa.ndfa_model_hyper_parameters import IdentifierEncoderParams
from ndfa.nn_utils.misc import get_activation_layer
from ndfa.nn_utils.vocabulary import Vocabulary
from ndfa.nn_utils.attn_rnn_encoder import AttnRNNEncoder
from ndfa.misc.tensors_data_class import BatchFlattenedSeq
from ndfa.code_nn_modules.code_task_input import IdentifiersInputTensors


__all__ = ['IdentifierEncoder']


# TODO: use `SequenceEncoder` instead of `AttnRNNEncoder`
class IdentifierEncoder(nn.Module):
    def __init__(self, sub_identifiers_vocab: Vocabulary,
                 encoder_params: IdentifierEncoderParams,
                 obfuscate_sub_parts: bool = True,
                 use_hashing_trick: bool = True,
                 nr_rnn_layers: int = 1,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(IdentifierEncoder, self).__init__()
        self.activation_layer = get_activation_layer(activation_fn)()
        self.sub_identifiers_vocab = sub_identifiers_vocab
        self.encoder_params = encoder_params
        self.obfuscate_sub_parts = obfuscate_sub_parts
        self.use_hashing_trick = use_hashing_trick

        if obfuscate_sub_parts:
            self.sub_identifiers_obfuscation_embedding_layer = nn.Embedding(
                num_embeddings=len(sub_identifiers_vocab), embedding_dim=self.encoder_params.identifier_embedding_dim,
                padding_idx=sub_identifiers_vocab.get_word_idx('<PAD>'))
        else:
            self.sub_identifiers_embedding_layer = nn.Embedding(
                num_embeddings=len(sub_identifiers_vocab), embedding_dim=self.encoder_params.identifier_embedding_dim,
                padding_idx=sub_identifiers_vocab.get_word_idx('<PAD>'))

        # TODO: use `SequenceEncoder` instead of `AttnRNNEncoder`
        self.attn_rnn_encoder = AttnRNNEncoder(
            input_dim=self.encoder_params.identifier_embedding_dim,
            hidden_dim=self.encoder_params.identifier_embedding_dim, rnn_type='lstm',
            nr_rnn_layers=nr_rnn_layers, rnn_bi_direction=True, activation_fn=activation_fn)

        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.identifier_sub_parts_hashing_linear = nn.Linear(
            in_features=self.encoder_params.nr_sub_identifier_hashing_features,
            out_features=self.encoder_params.nr_sub_identifier_hashing_features, bias=False)
        if use_hashing_trick:
            self.vocab_and_hashing_combiner = nn.Linear(
                in_features=self.encoder_params.identifier_embedding_dim +
                            self.encoder_params.nr_sub_identifier_hashing_features,
                out_features=self.encoder_params.identifier_embedding_dim +
                             self.encoder_params.nr_sub_identifier_hashing_features)
            self.vocab_and_hashing_combiner_final_projection_layer = nn.Linear(
                in_features=self.encoder_params.identifier_embedding_dim +
                            self.encoder_params.nr_sub_identifier_hashing_features,
                out_features=self.encoder_params.identifier_embedding_dim)

    def forward(self, identifiers: IdentifiersInputTensors):
        assert isinstance(identifiers.identifier_sub_parts_vocab_word_index, BatchFlattenedSeq)
        assert isinstance(identifiers.identifier_sub_parts_vocab_word_index.sequences, torch.Tensor)
        assert identifiers.identifier_sub_parts_hashings is None or \
               isinstance(identifiers.identifier_sub_parts_hashings, BatchFlattenedSeq)
        assert identifiers.identifier_sub_parts_hashings is None or \
               isinstance(identifiers.identifier_sub_parts_hashings.sequences, torch.Tensor)
        assert identifiers.identifier_sub_parts_vocab_word_index.sequences.dtype == torch.long
        assert identifiers.identifier_sub_parts_vocab_word_index.sequences.ndim == 2
        nr_identifiers_in_batch, max_nr_sub_identifiers_in_identifier = identifiers.identifier_sub_parts_vocab_word_index.sequences.size()

        if self.obfuscate_sub_parts:
            identifier_sub_parts_obfuscated_indices = identifiers.sub_parts_obfuscation.sample[
                identifiers.identifier_sub_parts_index.sequences]
            identifier_sub_parts_obfuscated_indices.masked_fill_(
                ~identifiers.identifier_sub_parts_index.sequences_mask, 0)
            identifiers_sub_parts_embeddings = self.sub_identifiers_obfuscation_embedding_layer(
                identifier_sub_parts_obfuscated_indices)
            identifiers_sub_parts_embeddings = self.dropout_layer(identifiers_sub_parts_embeddings)
        else:
            identifiers_sub_parts_embeddings = self.sub_identifiers_embedding_layer(
                identifiers.identifier_sub_parts_vocab_word_index.sequences)
            identifiers_sub_parts_embeddings = self.dropout_layer(identifiers_sub_parts_embeddings)
        assert identifiers_sub_parts_embeddings.size() == \
               (nr_identifiers_in_batch, max_nr_sub_identifiers_in_identifier, self.encoder_params.identifier_embedding_dim)

        if self.use_hashing_trick:
            assert identifiers.identifier_sub_parts_hashings is not None
            identifiers_sub_parts_hashings_projected = self.identifier_sub_parts_hashing_linear(
                identifiers.identifier_sub_parts_hashings.sequences)
            identifiers_sub_parts_hashings_projected = self.dropout_layer(identifiers_sub_parts_hashings_projected)
            identifiers_sub_parts_vocab_embeddings_and_hashings_combined = self.vocab_and_hashing_combiner(
                torch.cat([identifiers_sub_parts_embeddings, identifiers_sub_parts_hashings_projected], dim=-1))
            identifiers_sub_parts_embeddings = self.dropout_layer(self.activation_layer(
                identifiers_sub_parts_vocab_embeddings_and_hashings_combined))
            identifiers_sub_parts_embeddings = self.vocab_and_hashing_combiner_final_projection_layer(
                identifiers_sub_parts_embeddings)
            identifiers_sub_parts_embeddings = self.dropout_layer(self.activation_layer(
                identifiers_sub_parts_embeddings))
            identifiers_sub_parts_embeddings = identifiers_sub_parts_embeddings.masked_fill(
                ~identifiers.identifier_sub_parts_index.sequences_mask.unsqueeze(-1)
                    .expand(identifiers_sub_parts_embeddings.shape), 0)

        encoded_identifiers, _ = self.attn_rnn_encoder(
            sequence_input=identifiers_sub_parts_embeddings,
            lengths=identifiers.identifier_sub_parts_vocab_word_index.sequences_lengths,
            batch_first=True)
        assert encoded_identifiers.size() == (nr_identifiers_in_batch, self.encoder_params.identifier_embedding_dim)
        return encoded_identifiers

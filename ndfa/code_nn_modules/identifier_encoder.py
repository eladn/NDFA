import torch
import torch.nn as nn

from ndfa.ndfa_model_hyper_parameters import IdentifierEncoderParams
from ndfa.nn_utils.misc import get_activation_layer
from ndfa.nn_utils.vocabulary import Vocabulary
from ndfa.nn_utils.attn_rnn_encoder import AttnRNNEncoder
from ndfa.misc.tensors_data_class import BatchFlattenedSeq
from ndfa.code_nn_modules.code_task_input import IdentifiersInputTensors
from ndfa.nn_utils.embedding_with_obfuscation import EmbeddingWithObfuscation


__all__ = ['IdentifierEncoder']


# TODO: add `EmbeddingWithObfuscationParams` to `IdentifierEncoderParams`.
# TODO: use `SequenceEncoder` instead of `AttnRNNEncoder`
class IdentifierEncoder(nn.Module):
    def __init__(self, sub_identifiers_vocab: Vocabulary,
                 encoder_params: IdentifierEncoderParams,
                 sub_parts_obfuscation: str = 'replace_oov_and_random',
                 sub_parts_obfuscation_rate: float = 0.3,
                 use_vocab: bool = True,
                 use_hashing_trick: bool = True,
                 nr_rnn_layers: int = 1,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(IdentifierEncoder, self).__init__()
        assert sub_parts_obfuscation in {'none', 'add_all', 'replace_all', 'replace_oovs',
                                         'replace_random', 'replace_oov_and_random'}
        self.sub_identifiers_vocab = sub_identifiers_vocab
        self.encoder_params = encoder_params
        self.sub_parts_obfuscation = sub_parts_obfuscation
        self.use_hashing_trick = use_hashing_trick

        self.sub_identifiers_embedding = EmbeddingWithObfuscation(
            vocab=sub_identifiers_vocab, embedding_dim=self.encoder_params.identifier_embedding_dim,
            obfuscation_type=sub_parts_obfuscation, obfuscation_rate=sub_parts_obfuscation_rate,
            use_vocab=use_vocab, use_hashing_trick=use_hashing_trick,
            nr_hashing_features=self.encoder_params.nr_sub_identifier_hashing_features,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

        # TODO: use `SequenceEncoder` instead of `AttnRNNEncoder`
        self.attn_rnn_encoder = AttnRNNEncoder(
            input_dim=self.encoder_params.identifier_embedding_dim,
            hidden_dim=self.encoder_params.identifier_embedding_dim, rnn_type='lstm',
            nr_rnn_layers=nr_rnn_layers, rnn_bi_direction=True, activation_fn=activation_fn)

        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

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

        identifiers_sub_parts_embeddings = self.sub_identifiers_embedding(
            vocab_word_idx=identifiers.identifier_sub_parts_vocab_word_index.sequences,
            word_hashes=identifiers.identifier_sub_parts_hashings.sequences,
            batch_unique_word_idx=identifiers.identifier_sub_parts_index.sequences,
            obfuscation_vocab_random_indices_shuffle=identifiers.sub_parts_obfuscation.sample)
        identifiers_sub_parts_embeddings = identifiers_sub_parts_embeddings.masked_fill(
                ~identifiers.identifier_sub_parts_index.sequences_mask.unsqueeze(-1)
                    .expand(identifiers_sub_parts_embeddings.shape), 0)

        encoded_identifiers, _ = self.attn_rnn_encoder(
            sequence_input=identifiers_sub_parts_embeddings,
            lengths=identifiers.identifier_sub_parts_vocab_word_index.sequences_lengths,
            batch_first=True)
        assert encoded_identifiers.size() == (nr_identifiers_in_batch, self.encoder_params.identifier_embedding_dim)
        return encoded_identifiers

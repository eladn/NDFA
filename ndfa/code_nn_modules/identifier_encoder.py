import torch
import torch.nn as nn

from ndfa.code_nn_modules.params.identifier_encoder_params import IdentifierEncoderParams
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.nn_utils.modules.sequence_encoder import SequenceEncoder
from ndfa.nn_utils.modules.sequence_combiner import SequenceCombiner
from ndfa.misc.tensors_data_class import BatchFlattenedSeq
from ndfa.code_nn_modules.code_task_input import IdentifiersInputTensors
from ndfa.nn_utils.modules.embedding_with_unknowns import EmbeddingWithUnknowns


__all__ = ['IdentifierEncoder']


class IdentifierEncoder(nn.Module):
    def __init__(self,
                 identifiers_vocab: Vocabulary,
                 sub_identifiers_vocab: Vocabulary,
                 encoder_params: IdentifierEncoderParams,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(IdentifierEncoder, self).__init__()
        self.sub_identifiers_vocab = sub_identifiers_vocab
        self.encoder_params = encoder_params

        if self.encoder_params.use_sub_identifiers:
            self.sub_identifiers_embedding = EmbeddingWithUnknowns(
                vocab=sub_identifiers_vocab,
                embedding_dim=self.encoder_params.identifier_embedding_dim,
                embedding_params=self.encoder_params.embedding_params,
                nr_hashing_features=self.encoder_params.nr_sub_identifier_hashing_features,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            self.sequence_encoder = SequenceEncoder(
                encoder_params=self.encoder_params.sequence_encoder,
                input_dim=self.encoder_params.identifier_embedding_dim,
                hidden_dim=self.encoder_params.identifier_embedding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            self.sequence_combiner = SequenceCombiner(
                encoding_dim=self.encoder_params.identifier_embedding_dim,
                combined_dim=self.encoder_params.identifier_embedding_dim,
                combiner_params=self.encoder_params.sequence_combiner,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        else:
            self.identifiers_embedding = EmbeddingWithUnknowns(
                vocab=identifiers_vocab,
                embedding_dim=self.encoder_params.identifier_embedding_dim,
                embedding_params=self.encoder_params.embedding_params,
                nr_hashing_features=self.encoder_params.nr_identifier_hashing_features,
                dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(self, identifiers_input: IdentifiersInputTensors):
        if self.encoder_params.use_sub_identifiers:
            assert isinstance(identifiers_input.identifier_sub_parts_vocab_word_index, BatchFlattenedSeq)
            assert isinstance(identifiers_input.identifier_sub_parts_vocab_word_index.sequences, torch.Tensor)
            assert identifiers_input.identifier_sub_parts_hashings is None or \
                   isinstance(identifiers_input.identifier_sub_parts_hashings, BatchFlattenedSeq)
            assert identifiers_input.identifier_sub_parts_hashings is None or \
                   isinstance(identifiers_input.identifier_sub_parts_hashings.sequences, torch.Tensor)
            assert identifiers_input.identifier_sub_parts_vocab_word_index.sequences.dtype == torch.long
            assert identifiers_input.identifier_sub_parts_vocab_word_index.sequences.ndim == 2
            nr_identifiers_in_batch, max_nr_sub_identifiers_in_identifier = identifiers_input.identifier_sub_parts_vocab_word_index.sequences.size()

            identifiers_sub_parts_embeddings = self.sub_identifiers_embedding(
                vocab_word_idx=identifiers_input.identifier_sub_parts_vocab_word_index.sequences,
                word_hashes=identifiers_input.identifier_sub_parts_hashings.sequences
                if identifiers_input.identifier_sub_parts_hashings is not None else None,
                batch_unique_word_idx=identifiers_input.identifier_sub_parts_index.sequences,
                obfuscation_vocab_random_indices_shuffle_getter=lambda: identifiers_input.sub_parts_obfuscation.sample)
            identifiers_sub_parts_embeddings = identifiers_sub_parts_embeddings.masked_fill(
                    ~identifiers_input.identifier_sub_parts_index.sequences_mask.unsqueeze(-1)
                        .expand(identifiers_sub_parts_embeddings.shape), 0)

            encoded_identifiers_as_seq = self.sequence_encoder(
                sequence_input=identifiers_sub_parts_embeddings,
                lengths=identifiers_input.identifier_sub_parts_vocab_word_index.sequences_lengths,
                batch_first=True).sequence
            encoded_identifiers = self.sequence_combiner(
                sequence_encodings=encoded_identifiers_as_seq,
                sequence_lengths=identifiers_input.identifier_sub_parts_vocab_word_index.sequences_lengths,
                batch_first=True)
            assert encoded_identifiers.size() == (nr_identifiers_in_batch, self.encoder_params.identifier_embedding_dim)
            return encoded_identifiers
        else:
            nr_identifiers = identifiers_input.identifiers_vocab_word_index.tensor.size(0)
            return self.identifiers_embedding(
                vocab_word_idx=identifiers_input.identifiers_vocab_word_index.tensor,
                word_hashes=None,  # identifiers.identifier_hashings.tensor  # TODO
                batch_unique_word_idx=torch.arange(nr_identifiers),
                obfuscation_vocab_random_indices_shuffle_getter=
                lambda: identifiers_input.identifiers_obfuscation.sample)

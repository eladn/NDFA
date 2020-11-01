import torch
import torch.nn as nn
from warnings import warn
from typing import Optional

from ndfa.nn_utils.misc import get_activation_layer
from ndfa.nn_utils.vocabulary import Vocabulary


__all__ = ['EmbeddingWithObfuscation']


class EmbeddingWithObfuscation(nn.Module):
    def __init__(self, vocab: Vocabulary,
                 embedding_dim: int,
                 obfuscation_type: str = 'oov_and_random',
                 obfuscation_rate: float = 0.3,
                 obfuscation_embeddings_type: str = 'learnable',
                 nr_obfuscation_words: Optional[int] = None,
                 use_vocab: bool = True,
                 use_hashing_trick: bool = False,
                 nr_hashing_features: Optional[int] = None,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(EmbeddingWithObfuscation, self).__init__()
        assert obfuscation_type in {'none', 'all', 'oovs', 'random', 'oov_and_random'}
        assert obfuscation_embeddings_type in {'learnable', 'fix_orthogonal'}
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.obfuscation_type = obfuscation_type
        self.obfuscation_rate = obfuscation_rate
        self.obfuscation_embeddings_type = obfuscation_embeddings_type
        self.nr_obfuscation_words = len(vocab) if nr_obfuscation_words is None else nr_obfuscation_words
        self.use_hashing_trick = use_hashing_trick
        self.use_vocab = use_vocab
        self.nr_hashing_features = len(vocab) if nr_hashing_features is None else nr_hashing_features

        if self.obfuscation_type != 'none':
            if obfuscation_embeddings_type == 'learnable':
                self.obfuscation_embedding_layer = nn.Embedding(
                    num_embeddings=len(vocab), embedding_dim=embedding_dim,
                    padding_idx=vocab.get_word_idx('<PAD>'))
            elif obfuscation_embeddings_type == 'fix_orthogonal':
                # TODO: we might want to fix `nr_obfuscation_words` to be `min(nr_obfuscation_words, embedding_dim)`
                # self.nr_obfuscation_words = min(self.nr_obfuscation_words, self.embedding_dim)
                if nr_hashing_features is not None and nr_obfuscation_words != self.nr_obfuscation_words:
                    warn(f'The chosen `nr_hashing_features` ({nr_hashing_features}) '
                         f'is > `embedding_dim` ({self.embedding_dim}), '
                         f'but the obfuscation embeddings are set to be initiated orthogonally. '
                         f'Therefore `nr_hashing_features` will be {self.nr_hashing_features}.')
                self.obfuscation_fixed_embeddings = torch.empty(
                    (self.nr_obfuscation_words, embedding_dim),
                    dtype=torch.float, requires_grad=False)
                nn.init.orthogonal_(self.obfuscation_fixed_embeddings)
        if self.obfuscation_type != 'all':
            assert self.use_vocab or self.use_hashing_trick
            if self.use_vocab:
                self.vocab_embedding_layer = nn.Embedding(
                    num_embeddings=self.nr_obfuscation_words, embedding_dim=embedding_dim,
                    padding_idx=vocab.get_word_idx('<PAD>'))
            if self.use_hashing_trick:
                self.hashing_linear = nn.Linear(
                    in_features=self.encoder_params.nr_sub_identifier_hashing_features,
                    out_features=self.encoder_params.nr_sub_identifier_hashing_features, bias=False)
            if self.use_vocab and self.use_hashing_trick:
                self.vocab_and_hashing_combiner = nn.Linear(
                    in_features=self.embedding_dim + self.nr_hashing_features,
                    out_features=self.embedding_dim + self.nr_hashing_features)
                self.vocab_and_hashing_combiner_final_projection_layer = nn.Linear(
                    in_features=self.embedding_dim + self.nr_hashing_features,
                    out_features=self.embedding_dim)

        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self, vocab_word_idx: Optional[torch.LongTensor] = None,
                word_hashes: Optional[torch.LongTensor] = None,
                batch_unique_word_idx: Optional[torch.LongTensor] = None,
                obfuscation_vocab_random_indices_shuffle: Optional[torch.LongTensor] = None):
        input_words_shape = None
        if vocab_word_idx is not None:
            input_words_shape = vocab_word_idx.shape
        elif word_hashes is not None:
            input_words_shape = word_hashes.shape[:-1]
        elif batch_unique_word_idx is not None:
            input_words_shape = batch_unique_word_idx.shape
        assert input_words_shape is not None
        assert vocab_word_idx is None or input_words_shape == vocab_word_idx.shape
        assert word_hashes is None or input_words_shape == word_hashes[:-1].shape
        assert batch_unique_word_idx is None or input_words_shape == batch_unique_word_idx.shape

        if self.obfuscation_type != 'none':
            assert batch_unique_word_idx is not None
            assert obfuscation_vocab_random_indices_shuffle is not None
            words_obfuscated_indices = obfuscation_vocab_random_indices_shuffle[batch_unique_word_idx]
            words_obfuscated_indices = words_obfuscated_indices % self.nr_obfuscation_words
            if self.obfuscation_embeddings_type == 'learnable':
                obfuscation_words_embeddings = self.obfuscation_embedding_layer(words_obfuscated_indices)
            elif self.obfuscation_embeddings_type == 'fix_orthogonal':
                obfuscation_words_embeddings = self.obfuscation_fixed_embeddings[words_obfuscated_indices]
            obfuscation_words_embeddings = self.dropout_layer(obfuscation_words_embeddings)
        if self.obfuscation_type != 'all':
            assert self.use_vocab or self.use_hashing_trick
            if self.use_vocab:
                assert vocab_word_idx is not None
                vocab_words_embeddings = self.dropout_layer(self.vocab_embedding_layer(vocab_word_idx))
                non_obfuscated_words_embeddings = vocab_words_embeddings
            if self.use_hashing_trick:
                assert word_hashes is not None
                hashing_words_embeddings = self.dropout_layer(self.hashing_linear(word_hashes))
                non_obfuscated_words_embeddings = hashing_words_embeddings
            if self.use_vocab and self.use_hashing_trick:
                vocab_and_hashing_words_embeddings = self.vocab_and_hashing_combiner(
                    torch.cat([vocab_words_embeddings, hashing_words_embeddings], dim=-1))
                vocab_and_hashing_words_embeddings = self.dropout_layer(self.activation_layer(
                    vocab_and_hashing_words_embeddings))
                vocab_and_hashing_words_embeddings = self.vocab_and_hashing_combiner_final_projection_layer(
                    vocab_and_hashing_words_embeddings)
                vocab_and_hashing_words_embeddings = self.dropout_layer(self.activation_layer(
                    vocab_and_hashing_words_embeddings))
                non_obfuscated_words_embeddings = vocab_and_hashing_words_embeddings

        if self.obfuscation_type == 'all':
            pass
        elif self.obfuscation_type == 'oovs':
            pass
        elif self.obfuscation_type == 'random':
            pass
        elif self.obfuscation_type == 'oov_and_random':
            pass

        raise NotImplementedError

        assert final_words_embeddings.size() == input_words_shape + (self.embedding_dim,)
        return final_words_embeddings

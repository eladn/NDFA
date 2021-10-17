import torch
import torch.nn as nn
from warnings import warn
from typing import Optional

from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.nn_utils.modules.params.embedding_with_unknowns_params import EmbeddingWithUnknownsParams


__all__ = ['EmbeddingWithUnknowns']


class EmbeddingWithUnknowns(nn.Module):
    def __init__(self, vocab: Vocabulary,
                 embedding_dim: int,
                 embedding_params: EmbeddingWithUnknownsParams,
                 nr_hashing_features: Optional[int] = None,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(EmbeddingWithUnknowns, self).__init__()
        self.embedding_params = embedding_params
        assert self.embedding_params.obfuscation_type in \
               {'none', 'add_all', 'replace_all', 'replace_oovs',
                'replace_random', 'replace_oov_and_random'}
        assert self.embedding_params.obfuscation_embeddings_type in \
               {'learnable', 'fixed_orthogonal'}
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.nr_obfuscation_words = \
            len(vocab) if embedding_params.nr_obfuscation_words is None else embedding_params.nr_obfuscation_words
        self.use_hashing_trick = False if self.embedding_params.obfuscation_type == 'replace_all' else \
            self.embedding_params.use_hashing_trick
        if self.embedding_params.use_hashing_trick and self.embedding_params.obfuscation_type == 'replace_all':
            warn('`use_hashing_trick` is set on, but `obfuscation_type` is set to `replace_all`.')
        self.use_vocab = False if self.embedding_params.obfuscation_type == 'replace_all' else \
            self.embedding_params.use_vocab
        if self.embedding_params.use_vocab and self.embedding_params.obfuscation_type == 'replace_all':
            warn('`use_vocab` is set on, but `obfuscation_type` is set to `replace_all`.')
        self.nr_hashing_features = len(vocab) if nr_hashing_features is None else nr_hashing_features

        if self.embedding_params.obfuscation_type != 'none':
            if self.embedding_params.obfuscation_embeddings_type == 'learnable':
                # Note: we don't set here `padding_idx=vocab.get_word_idx('<PAD>')`, because
                # all of the words here are accessed randomly. We mask-out the paddings later.
                self.obfuscation_embedding_layer = nn.Embedding(
                    num_embeddings=self.nr_obfuscation_words, embedding_dim=self.embedding_dim)
            elif self.embedding_params.obfuscation_embeddings_type == 'fixed_orthogonal':
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
        if self.embedding_params.obfuscation_type != 'replace_all':
            if not self.use_vocab and not self.use_hashing_trick:
                raise ValueError(f'`obfuscation_type` is set to `{self.embedding_params.obfuscation_type}` '
                                 f'(!= `replace_all`), but neither `use_vocab` nor `use_hashing_trick` is set.')
            if self.use_vocab:
                self.vocab_embedding_layer = nn.Embedding(
                    num_embeddings=len(vocab), embedding_dim=embedding_dim,
                    padding_idx=vocab.get_word_idx('<PAD>'))
            if self.use_hashing_trick:
                self.hashing_linear = nn.Linear(
                    in_features=self.nr_hashing_features,
                    out_features=self.nr_hashing_features, bias=False)
            if int(self.use_vocab) + int(self.use_hashing_trick) + \
                    int(self.embedding_params.obfuscation_type == 'add_all') >= 2:
                combiner_input_dim = 0
                if self.use_vocab:
                    combiner_input_dim += self.embedding_dim
                if self.use_hashing_trick:
                    combiner_input_dim += self.nr_hashing_features
                if self.embedding_params.obfuscation_type == 'add_all':
                    combiner_input_dim += self.embedding_dim
                self.combiner_first_layer = nn.Linear(
                    in_features=combiner_input_dim, out_features=combiner_input_dim)
                self.combiner_second_layer = nn.Linear(
                    in_features=combiner_input_dim, out_features=self.embedding_dim)

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
        assert word_hashes is None or input_words_shape == word_hashes.shape[:-1]
        assert batch_unique_word_idx is None or input_words_shape == batch_unique_word_idx.shape
        output_shape = input_words_shape + (self.embedding_dim,)

        pad_mask = None
        if vocab_word_idx is not None:
            pad_mask = (vocab_word_idx == self.vocab.get_word_idx('<PAD>'))
            pad_mask = pad_mask.unsqueeze(-1).expand(output_shape)

        if self.embedding_params.obfuscation_type != 'none':
            assert batch_unique_word_idx is not None
            assert obfuscation_vocab_random_indices_shuffle is not None
            words_obfuscated_indices = obfuscation_vocab_random_indices_shuffle[batch_unique_word_idx]
            words_obfuscated_indices = words_obfuscated_indices % self.nr_obfuscation_words
            if self.embedding_params.obfuscation_embeddings_type == 'learnable':
                obfuscation_words_embeddings = self.obfuscation_embedding_layer(words_obfuscated_indices)
            elif self.embedding_params.obfuscation_embeddings_type == 'fixed_orthogonal':
                obfuscation_words_embeddings = self.obfuscation_fixed_embeddings[words_obfuscated_indices]
            obfuscation_words_embeddings = self.dropout_layer(obfuscation_words_embeddings)

        if self.embedding_params.obfuscation_type == 'replace_all':
            final_words_embeddings = obfuscation_words_embeddings
            if pad_mask is not None:
                final_words_embeddings = final_words_embeddings.masked_fill(pad_mask, 0)
            assert final_words_embeddings.shape == output_shape
            return final_words_embeddings

        assert self.use_vocab or self.use_hashing_trick
        if self.use_vocab:
            assert vocab_word_idx is not None
            vocab_words_embeddings = self.dropout_layer(self.vocab_embedding_layer(vocab_word_idx))
            non_obfuscated_words_embeddings = vocab_words_embeddings
        if self.use_hashing_trick:
            assert word_hashes is not None
            assert word_hashes.size(-1) == self.nr_hashing_features
            hashing_words_embeddings = self.dropout_layer(self.hashing_linear(word_hashes))
            non_obfuscated_words_embeddings = hashing_words_embeddings

        if int(self.use_vocab) + int(self.use_hashing_trick) + \
                int(self.embedding_params.obfuscation_type == 'add_all') >= 2:
            combiner_inputs = []
            if self.use_vocab:
                combiner_inputs.append(vocab_words_embeddings)
            if self.use_hashing_trick:
                combiner_inputs.append(hashing_words_embeddings)
            if self.embedding_params.obfuscation_type == 'add_all':
                combiner_inputs.append(obfuscation_words_embeddings)
            combined_words_embeddings = self.combiner_first_layer(torch.cat(combiner_inputs, dim=-1))
            combined_words_embeddings = self.dropout_layer(self.activation_layer(combined_words_embeddings))
            combined_words_embeddings = self.dropout_layer(self.combiner_second_layer(combined_words_embeddings))
            if self.embedding_params.obfuscation_type != 'add_all':
                non_obfuscated_words_embeddings = combined_words_embeddings

        if self.embedding_params.obfuscation_type in {'replace_oovs', 'replace_oov_and_random'}:
            assert vocab_word_idx is not None  # we need it to detect the OOVs.
            oovs_mask = (vocab_word_idx == self.vocab.get_word_idx('<UNK>'))
            oovs_mask = oovs_mask.unsqueeze(-1).expand(obfuscation_words_embeddings.shape)
            words_with_oov_obfuscated_embeddings = torch.where(
                oovs_mask, obfuscation_words_embeddings, non_obfuscated_words_embeddings)

        if (self.training or self.embedding_params.replace_random_in_inference) and \
                self.embedding_params.obfuscation_type in {'replace_random', 'replace_oov_and_random'}:
            # TODO: consider using a given dedicated RNG here.
            random_obfuscation_probs = torch.rand(input_words_shape, device=obfuscation_words_embeddings.device)
            random_obfuscation_mask = (random_obfuscation_probs < self.embedding_params.obfuscation_rate)
            random_obfuscation_mask = random_obfuscation_mask.unsqueeze(-1).expand(obfuscation_words_embeddings.shape)

        if self.embedding_params.obfuscation_type == 'none':
            final_words_embeddings = non_obfuscated_words_embeddings
        elif self.embedding_params.obfuscation_type == 'add_all':
            final_words_embeddings = combined_words_embeddings
        elif self.embedding_params.obfuscation_type == 'replace_oovs':
            final_words_embeddings = words_with_oov_obfuscated_embeddings
        elif self.embedding_params.obfuscation_type == 'replace_random':
            final_words_embeddings = non_obfuscated_words_embeddings
            if self.training or self.embedding_params.replace_random_in_inference:
                final_words_embeddings = torch.where(
                    random_obfuscation_mask, obfuscation_words_embeddings, final_words_embeddings)
        elif self.embedding_params.obfuscation_type == 'replace_oov_and_random':
            final_words_embeddings = words_with_oov_obfuscated_embeddings
            if self.training or self.embedding_params.replace_random_in_inference:
                final_words_embeddings = torch.where(
                    random_obfuscation_mask, obfuscation_words_embeddings, final_words_embeddings)

        if pad_mask is not None:
            final_words_embeddings = final_words_embeddings.masked_fill(pad_mask, 0)
        assert final_words_embeddings.shape == output_shape
        return final_words_embeddings

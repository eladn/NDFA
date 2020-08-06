import torch
import torch.nn as nn
import torch.nn.functional as F

from ndfa.nn_utils.apply_batched_embeddings import apply_batched_embeddings
from ndfa.code_nn_modules.vocabulary import Vocabulary


class SymbolsEncoder(nn.Module):
    def __init__(self, symbols_special_words_vocab: Vocabulary, symbol_embedding_dim: int):
        super(SymbolsEncoder, self).__init__()
        self.symbols_special_words_vocab = symbols_special_words_vocab
        self.symbol_embedding_dim = symbol_embedding_dim
        # FIXME: might be problematic because 2 different modules hold `symbols_special_words_embedding` (both SymbolsEncoder and SymbolsDecoder).
        self.symbols_special_words_embedding = nn.Embedding(
            num_embeddings=len(self.symbols_special_words_vocab),
            embedding_dim=symbol_embedding_dim,
            padding_idx=self.symbols_special_words_vocab.get_word_idx('<PAD>'))

    def forward(self, encoded_identifiers, identifiers_idxs_of_all_symbols, identifiers_idxs_of_all_symbols_mask):
        assert identifiers_idxs_of_all_symbols.size()[0] \
               == identifiers_idxs_of_all_symbols_mask.size()[0] \
               == encoded_identifiers.size()[0]
        assert encoded_identifiers.size()[-1] == self.symbol_embedding_dim  # it might change...
        symbol_pad_embed = self.symbols_special_words_embedding(
            torch.tensor([self.symbols_special_words_vocab.get_word_idx('<PAD>')],
                         dtype=torch.long, device=encoded_identifiers.device)).view(-1)
        symbols_encodings = apply_batched_embeddings(
            batched_embeddings=encoded_identifiers,
            indices=identifiers_idxs_of_all_symbols,
            mask=identifiers_idxs_of_all_symbols_mask,
            padding_embedding_vector=symbol_pad_embed)  # (batch_size, nr_symbols, symbol_embedding_dim)
        assert symbols_encodings.size() == identifiers_idxs_of_all_symbols.size() + (self.symbol_embedding_dim,)
        return symbols_encodings

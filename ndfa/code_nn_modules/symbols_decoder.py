import torch
import torch.nn as nn
import typing

from ndfa.nn_utils.attn_rnn_decoder import AttnRNNDecoder
from ndfa.nn_utils.vocabulary import Vocabulary
from ndfa.nn_utils.scattered_encodings import ScatteredEncodings
from ndfa.code_nn_modules.code_task_input import SymbolsInputTensors


class SymbolsDecoder(nn.Module):
    def __init__(self, symbols_special_words_embedding: nn.Embedding, symbols_special_words_vocab: Vocabulary,
                 max_nr_taget_symbols: int, encoder_output_dim: int = 256, symbols_encoding_dim: int = 256,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu',
                 use_batch_flattened_target_symbols_vocab: bool = False):
        super(SymbolsDecoder, self).__init__()
        self.use_batch_flattened_target_symbols_vocab = use_batch_flattened_target_symbols_vocab
        # FIXME: might be problematic because 2 different modules hold `symbols_special_words_embedding` (both SymbolsEncoder and SymbolsDecoder).
        self.attn_rnn_decoder = AttnRNNDecoder(
            encoder_output_dim=encoder_output_dim,
            max_target_seq_len=max_nr_taget_symbols, decoder_hidden_dim=max(symbols_encoding_dim * 4, encoder_output_dim),
            decoder_output_dim=symbols_encoding_dim, embedding_dropout_rate=dropout_rate,
            activation_fn=activation_fn, rnn_type='lstm', nr_rnn_layers=2,
            output_common_embedding=symbols_special_words_embedding, output_common_vocab=symbols_special_words_vocab)

    def forward(self, encoder_outputs: torch.Tensor,
                encoder_outputs_mask: typing.Optional[torch.BoolTensor],
                symbols: SymbolsInputTensors,
                batched_flattened_symbols_encodings: torch.Tensor,
                encoded_symbols_occurrences: typing.Optional[ScatteredEncodings] = None,
                groundtruth_target_symbols_idxs: typing.Optional[torch.LongTensor] = None):
        output_vocab_batch_flattened_encodings = None
        output_vocab_example_based_encodings = None
        output_vocab_example_based_encodings_mask = None
        if self.use_batch_flattened_target_symbols_vocab and self.training:
            output_vocab_batch_flattened_encodings = batched_flattened_symbols_encodings
        else:
            output_vocab_example_based_encodings = \
                symbols.symbols_identifier_indices.unflatten(batched_flattened_symbols_encodings)
            assert output_vocab_example_based_encodings.ndim == 3
            output_vocab_example_based_encodings_mask = symbols.symbols_identifier_indices.unflattener_mask

        return self.attn_rnn_decoder(
            encoder_outputs=encoder_outputs, encoder_outputs_mask=encoder_outputs_mask,
            output_vocab_batch_flattened_encodings=output_vocab_batch_flattened_encodings,
            output_vocab_example_based_encodings=output_vocab_example_based_encodings,
            output_vocab_example_based_encodings_mask=output_vocab_example_based_encodings_mask,
            dyn_vocab_scattered_encodings=encoded_symbols_occurrences,
            groundtruth_target_idxs=groundtruth_target_symbols_idxs)

from typing import NamedTuple

from ndfa.ndfa_model_hyper_parameters import NDFAModelHyperParams
from ndfa.misc.code_data_structure_api import *
from ndfa.misc.iter_raw_extracted_data_files import iter_raw_extracted_examples_and_verify
from ndfa.code_nn_modules.vocabulary import Vocabulary


__all__ = ['CodeTaskVocabs', 'kos_token_to_kos_token_vocab_word']


# TODO: put in utils
def get_pdg_node_tokenized_expression(method: SerMethod, pdg_node: SerPDGNode):
    return method.code.tokenized[
        pdg_node.code_sub_token_range_ref.begin_token_idx:
        pdg_node.code_sub_token_range_ref.end_token_idx+1]


class CodeTaskVocabs(NamedTuple):
    sub_identifiers: Vocabulary
    kos_tokens: Vocabulary
    pdg_node_control_kinds: Vocabulary
    tokens_kinds: Vocabulary
    pdg_control_flow_edge_types: Vocabulary
    symbols_special_words: Vocabulary
    expressions_special_words: Vocabulary
    identifiers_special_words: Vocabulary

    @classmethod
    def load_or_create(cls, model_hps: NDFAModelHyperParams, pp_data_path: str,
                       raw_train_data_path: Optional[str] = None) -> 'CodeTaskVocabs':
        print('Loading / creating code task vocabularies ..')
        vocabs_pad_unk_special_words = ('<PAD>', '<UNK>')

        sub_identifiers_carpus_generator = None if raw_train_data_path is None else lambda: (
            sub_identifier
            for example in iter_raw_extracted_examples_and_verify(raw_extracted_data_dir=raw_train_data_path)
            for identifier_as_sub_identifiers in example.method_pdg.sub_identifiers_by_idx
            for sub_identifier in identifier_as_sub_identifiers)
        sub_identifiers_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='sub_identifiers',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words + ('<EOI>',), min_word_freq=40,
            max_vocab_size_wo_specials=1000, carpus_generator=sub_identifiers_carpus_generator)

        kos_tokens_carpus_generator = None if raw_train_data_path is None else lambda: (
            kos_token_to_kos_token_vocab_word(token)
            for example in iter_raw_extracted_examples_and_verify(raw_extracted_data_dir=raw_train_data_path)
            for pdg_node in example.method_pdg.pdg_nodes
            if pdg_node.code_sub_token_range_ref is not None
            for token in get_pdg_node_tokenized_expression(example.method, pdg_node)
            if token.kind in {SerTokenKind.KEYWORD, SerTokenKind.OPERATOR, SerTokenKind.SEPARATOR})
        kos_tokens_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='kos_tokens',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words + ('<NONE>',), min_word_freq=200,
            carpus_generator=kos_tokens_carpus_generator)

        pdg_node_control_kinds_carpus_generator = None if raw_train_data_path is None else lambda: (
            pdg_node.control_kind.value
            for example in iter_raw_extracted_examples_and_verify(raw_extracted_data_dir=raw_train_data_path)
            for pdg_node in example.method_pdg.pdg_nodes)
        # TODO: '<LOG_PRED>' special word is specific to LoggingCalls task. make it more generic here.
        pdg_node_control_kinds_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='pdg_node_control_kinds',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words + ('<LOG_PRED>',), min_word_freq=200,
            carpus_generator=pdg_node_control_kinds_carpus_generator)

        tokens_kinds_carpus_generator = None if raw_train_data_path is None else lambda: (
            token.kind.value
            for example in iter_raw_extracted_examples_and_verify(raw_extracted_data_dir=raw_train_data_path)
            for pdg_node in example.method_pdg.pdg_nodes
            if pdg_node.code_sub_token_range_ref is not None
            for token in get_pdg_node_tokenized_expression(example.method, pdg_node))
        tokens_kinds_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='tokens_kinds',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words, min_word_freq=200,
            carpus_generator=tokens_kinds_carpus_generator)

        pdg_control_flow_edge_types_carpus_generator = None if raw_train_data_path is None else lambda: (
            edge.type.value
            for example in iter_raw_extracted_examples_and_verify(raw_extracted_data_dir=raw_train_data_path)
            for pdg_node in example.method_pdg.pdg_nodes
            for edge in pdg_node.control_flow_out_edges)
        pdg_control_flow_edge_types_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='pdg_control_flow_edge_types',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words, min_word_freq=200,
            carpus_generator=pdg_control_flow_edge_types_carpus_generator)

        symbols_special_words_vocab = Vocabulary(
            name='symbols-specials', all_words_sorted_by_idx=[], params=(),
            special_words_sorted_by_idx=('<PAD>', '<SOS>', '<EOS>'))

        expressions_special_words_vocab = Vocabulary(
            name='expressions-specials', all_words_sorted_by_idx=[], params=(),
            special_words_sorted_by_idx=('<NONE>',))

        identifiers_special_words_vocab = Vocabulary(
            name='identifiers-specials', all_words_sorted_by_idx=[], params=(),
            special_words_sorted_by_idx=('<NONE>',))

        print('Done loading / creating vocabularies.')

        return CodeTaskVocabs(
            sub_identifiers=sub_identifiers_vocab,
            kos_tokens=kos_tokens_vocab,
            pdg_node_control_kinds=pdg_node_control_kinds_vocab,
            tokens_kinds=tokens_kinds_vocab,
            pdg_control_flow_edge_types=pdg_control_flow_edge_types_vocab,
            symbols_special_words=symbols_special_words_vocab,
            expressions_special_words=expressions_special_words_vocab,
            identifiers_special_words=identifiers_special_words_vocab)


def kos_token_to_kos_token_vocab_word(token: SerToken):
    assert token.kind in {SerTokenKind.KEYWORD, SerTokenKind.OPERATOR, SerTokenKind.SEPARATOR}
    if token.kind == SerTokenKind.KEYWORD:
        return f'kwrd_{token.text}'
    elif token.kind == SerTokenKind.OPERATOR:
        return f'op_{token.operator.value}'
    elif token.kind == SerTokenKind.SEPARATOR:
        return f'sep_{token.separator.value}'

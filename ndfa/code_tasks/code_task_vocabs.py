from typing import NamedTuple, Optional

from ndfa.ndfa_model_hyper_parameters import NDFAModelHyperParams
from ndfa.misc.code_data_structure_api import SerTokenKind, SerToken, SerASTNodeType
from ndfa.misc.code_data_structure_utils import get_pdg_node_tokenized_expression
from ndfa.misc.iter_raw_extracted_data_files import iter_raw_extracted_examples_and_verify
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary


__all__ = ['CodeTaskVocabs', 'kos_token_to_kos_token_vocab_word']


class CodeTaskVocabs(NamedTuple):
    identifiers: Vocabulary
    sub_identifiers: Vocabulary
    kos_tokens: Vocabulary
    pdg_node_control_kinds: Vocabulary
    tokens_kinds: Vocabulary
    pdg_control_flow_edge_types: Vocabulary
    ast_node_types: Vocabulary
    ast_node_major_types: Vocabulary
    ast_node_minor_types: Vocabulary
    primitive_types: Vocabulary
    modifiers: Vocabulary
    ast_traversal_orientation: Vocabulary
    ast_node_nr_children: Vocabulary
    ast_node_child_pos: Vocabulary
    symbols_special_words: Vocabulary
    expressions_special_words: Vocabulary
    identifiers_special_words: Vocabulary

    @classmethod
    def load_or_create(cls, model_hps: NDFAModelHyperParams, pp_data_path: str,
                       raw_train_data_path: Optional[str] = None) -> 'CodeTaskVocabs':
        print('Loading / creating code task vocabularies ..')
        vocabs_pad_unk_special_words = ('<PAD>', '<UNK>')

        identifiers_carpus_generator = None if raw_train_data_path is None else lambda: (
            identifier
            for example in iter_raw_extracted_examples_and_verify(
                raw_extracted_data_dir=raw_train_data_path, show_progress_bar=True)
            for identifier in example.method_pdg.identifier_by_idx)
        identifiers_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='identifiers',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words + ('<EOI>',), min_word_freq=40,
            max_vocab_size_wo_specials=model_hps.method_code_encoder.max_sub_identifier_vocab_size,
            carpus_generator=identifiers_carpus_generator)

        sub_identifiers_carpus_generator = None if raw_train_data_path is None else lambda: (
            sub_identifier
            for example in iter_raw_extracted_examples_and_verify(
                raw_extracted_data_dir=raw_train_data_path, show_progress_bar=True)
            for identifier_as_sub_identifiers in example.method_pdg.sub_identifiers_by_idx
            for sub_identifier in identifier_as_sub_identifiers)
        sub_identifiers_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='sub_identifiers',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words + ('<EOI>',), min_word_freq=40,
            max_vocab_size_wo_specials=model_hps.method_code_encoder.max_sub_identifier_vocab_size,
            carpus_generator=sub_identifiers_carpus_generator)

        kos_tokens_carpus_generator = None if raw_train_data_path is None else lambda: (
            kos_token_to_kos_token_vocab_word(token)
            for example in iter_raw_extracted_examples_and_verify(
                raw_extracted_data_dir=raw_train_data_path, show_progress_bar=True)
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
            for example in iter_raw_extracted_examples_and_verify(
                raw_extracted_data_dir=raw_train_data_path, show_progress_bar=True)
            for pdg_node in example.method_pdg.pdg_nodes)
        # TODO: '<LOG_PRED>' special word is specific to LoggingCalls task. make it more generic here.
        pdg_node_control_kinds_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='pdg_node_control_kinds',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words + ('<LOG_PRED>',), min_word_freq=200,
            carpus_generator=pdg_node_control_kinds_carpus_generator)

        tokens_kinds_carpus_generator = None if raw_train_data_path is None else lambda: (
            token.kind.value
            for example in iter_raw_extracted_examples_and_verify(
                raw_extracted_data_dir=raw_train_data_path, show_progress_bar=True)
            for pdg_node in example.method_pdg.pdg_nodes
            if pdg_node.code_sub_token_range_ref is not None
            for token in get_pdg_node_tokenized_expression(example.method, pdg_node))
        # TODO: '<LOG_PRED>' special word is specific to LoggingCalls task. make it more generic here.
        tokens_kinds_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='tokens_kinds',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words + ('<LOG_PRED>',), min_word_freq=200,
            carpus_generator=tokens_kinds_carpus_generator)

        # FIXME: It actually contains 'DataDependency' edge type, but the vocab is called `control_flow_edge`.
        pdg_control_flow_edge_types_carpus_generator = None if raw_train_data_path is None else lambda: (
            edge.type.value
            for example in iter_raw_extracted_examples_and_verify(
                raw_extracted_data_dir=raw_train_data_path, show_progress_bar=True)
            for pdg_node in example.method_pdg.pdg_nodes
            for edge in pdg_node.control_flow_out_edges)
        pdg_control_flow_edge_types_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='pdg_control_flow_edge_types',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words + ('DataDependency',), min_word_freq=200,
            carpus_generator=pdg_control_flow_edge_types_carpus_generator)

        ast_node_types_carpus_generator = None if raw_train_data_path is None else lambda: (
            ast_node.type.value
            for example in iter_raw_extracted_examples_and_verify(
                raw_extracted_data_dir=raw_train_data_path, show_progress_bar=True)
            for ast_node in example.method_ast.nodes)
        ast_node_types_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='ast_node_types',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words + ('<LOG_PRED>',), min_word_freq=200,
            carpus_generator=ast_node_types_carpus_generator)

        ast_node_major_types_carpus_generator = None if raw_train_data_path is None else lambda: (
            ast_node.type.value.split('_')[0]
            for example in iter_raw_extracted_examples_and_verify(
                raw_extracted_data_dir=raw_train_data_path, show_progress_bar=True)
            for ast_node in example.method_ast.nodes)
        ast_node_major_types_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='ast_node_major_types',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words + ('<LOG_PRED>',), min_word_freq=200,
            carpus_generator=ast_node_major_types_carpus_generator)

        ast_node_minor_types_carpus_generator = None if raw_train_data_path is None else lambda: (
            ast_node.type.value[ast_node.type.value.find('_') + 1:]
            for example in iter_raw_extracted_examples_and_verify(
                raw_extracted_data_dir=raw_train_data_path, show_progress_bar=True)
            for ast_node in example.method_ast.nodes
            if '_' in ast_node.type.value)
        ast_node_minor_types_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='ast_node_minor_types',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words, min_word_freq=200,
            carpus_generator=ast_node_minor_types_carpus_generator)

        primitive_types_carpus_generator = None if raw_train_data_path is None else lambda: (
            ast_node.type_name
            for example in iter_raw_extracted_examples_and_verify(
                raw_extracted_data_dir=raw_train_data_path, show_progress_bar=True)
            for ast_node in example.method_ast.nodes
            if ast_node.type == SerASTNodeType.PRIMITIVE_TYPE and ast_node.type_name is not None)
        primitive_types_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='primitive_types',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words, min_word_freq=200,
            carpus_generator=primitive_types_carpus_generator)

        modifiers_carpus_generator = None if raw_train_data_path is None else lambda: (
            ast_node.modifier
            for example in iter_raw_extracted_examples_and_verify(
                raw_extracted_data_dir=raw_train_data_path, show_progress_bar=True)
            for ast_node in example.method_ast.nodes
            if ast_node.modifier is not None)
        modifiers_vocab = Vocabulary.load_or_create(
            preprocessed_data_dir_path=pp_data_path, vocab_name='modifiers',
            special_words_sorted_by_idx=vocabs_pad_unk_special_words, min_word_freq=200,
            carpus_generator=modifiers_carpus_generator)

        MAX_AST_NODE_NR_CHILDREN_TO_COUNT = 5
        ast_traversal_orientation_vocab = Vocabulary(
            name='ast_traversal_orientations',
            all_words_sorted_by_idx=['DIR=UP', 'DIR=DOWN', 'DIR=COMMON'] +
                                    ['child_place=UNK'] +
                                    [f'child_place={place}' for place in range(MAX_AST_NODE_NR_CHILDREN_TO_COUNT)],
            special_words_sorted_by_idx=('<PAD>',))

        ast_node_nr_children_vocab = Vocabulary(
            name='ast_node_nr_children',
            all_words_sorted_by_idx=[
                f'{nr_children}' for nr_children in range(0, MAX_AST_NODE_NR_CHILDREN_TO_COUNT + 1)],
            special_words_sorted_by_idx=('<PAD>', '<MORE>'))

        ast_node_child_pos_vocab = Vocabulary(
            name='ast_node_child_pos',
            all_words_sorted_by_idx=[f'+{place}' for place in range(1, MAX_AST_NODE_NR_CHILDREN_TO_COUNT + 1)] +
                                    [f'-{place}' for place in range(1, MAX_AST_NODE_NR_CHILDREN_TO_COUNT + 1)],
            special_words_sorted_by_idx=('<PAD>', '<+MORE>', '<-MORE>', '<+ROOT>', '<-ROOT>'))

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
            identifiers=identifiers_vocab,
            sub_identifiers=sub_identifiers_vocab,
            kos_tokens=kos_tokens_vocab,
            pdg_node_control_kinds=pdg_node_control_kinds_vocab,
            tokens_kinds=tokens_kinds_vocab,
            pdg_control_flow_edge_types=pdg_control_flow_edge_types_vocab,
            ast_node_types=ast_node_types_vocab,
            ast_node_major_types=ast_node_major_types_vocab,
            ast_node_minor_types=ast_node_minor_types_vocab,
            primitive_types=primitive_types_vocab,
            modifiers=modifiers_vocab,
            ast_traversal_orientation=ast_traversal_orientation_vocab,
            ast_node_nr_children=ast_node_nr_children_vocab,
            ast_node_child_pos=ast_node_child_pos_vocab,
            symbols_special_words=symbols_special_words_vocab,
            expressions_special_words=expressions_special_words_vocab,
            identifiers_special_words=identifiers_special_words_vocab)


def kos_token_to_kos_token_vocab_word(token: SerToken) -> str:
    assert token.kind in {SerTokenKind.KEYWORD, SerTokenKind.OPERATOR, SerTokenKind.SEPARATOR}
    if token.kind == SerTokenKind.KEYWORD:
        return f'kwrd_{token.text}'
    elif token.kind == SerTokenKind.OPERATOR:
        return f'op_{token.operator.value}'
    elif token.kind == SerTokenKind.SEPARATOR:
        return f'sep_{token.separator.value}'

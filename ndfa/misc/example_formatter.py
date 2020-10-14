import io
import functools
import collections

from ndfa.misc.iter_raw_extracted_data_files import RawExtractedExample


__all__ = ['format_example']


def format_example(example: RawExtractedExample) -> str:
    out_stream = io.StringIO()
    oprint = functools.partial(print, file=out_stream)

    if example.logging_call is not None:
        oprint('-' * 50)
        oprint(
            f'[{example.logging_call.hash}] {example.method.code_filepath}:{example.logging_call.get_formatted_log_lines()}')
        oprint(example.logging_call.code.code_str)
        oprint('-' * 50)

    token_range_start_markers = collections.defaultdict(list)
    token_range_end_markers = collections.defaultdict(list)
    for pdg_node in example.method_pdg.pdg_nodes:
        if pdg_node.code_sub_token_range_ref is None:
            continue
        token_range_start_markers[pdg_node.code_sub_token_range_ref.begin_token_idx].append(pdg_node)
        token_range_end_markers[pdg_node.code_sub_token_range_ref.end_token_idx].append(pdg_node)

    for token_idx, (token, next_token) in enumerate(zip(example.method.code.tokenized, example.method.code.tokenized[1:] + [None])):
        if token_idx in token_range_start_markers:
            oprint(f'[@{token_range_start_markers[token_idx][0].idx} ', end='')
        begin_char_idx = token.position_range_in_code_snippet_str.begin_idx
        end_char_idx = token.position_range_in_code_snippet_str.end_idx
        oprint(example.method.code.code_str[begin_char_idx:end_char_idx], end='')
        if token.identifier_idx is not None:
            oprint(f'<idn:{token.identifier_idx}>', end='')
        if token.symbol_idx is not None:
            oprint(f'<sym:{token.symbol_idx}>', end='')
        if token_idx in token_range_end_markers:
            oprint(f' ] ', end='')

        if next_token is not None:
            next_token_begin_char_idx = next_token.position_range_in_code_snippet_str.begin_idx
            oprint(example.method.code.code_str[end_char_idx:next_token_begin_char_idx], end='')
    oprint()
    oprint('-' * 50)

    return out_stream.getvalue()

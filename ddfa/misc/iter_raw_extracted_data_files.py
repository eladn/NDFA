import os
import json
from tqdm import tqdm
import random
import typing
import argparse
import dataclasses
from warnings import warn
from itertools import takewhile, repeat

from ddfa.misc.code_data_structure_api import *


__all__ = ['RawExtractedExample', 'iter_raw_extracted_examples']


@dataclasses.dataclass
class RawExtractedExample:
    logging_call_hash: typing.Optional[str] = None
    logging_call: typing.Optional[SerLoggingCall] = None
    method: typing.Optional[SerMethod] = None
    method_ast: typing.Optional[SerMethodAST] = None
    method_pdg: typing.Optional[SerMethodPDG] = None


@dataclasses.dataclass
class RawExtractedDataFiles:
    logging_call_hash: typing.Optional[typing.Any] = None
    logging_call: typing.Optional[typing.Any] = None
    method: typing.Optional[typing.Any] = None
    method_ast: typing.Optional[typing.Any] = None
    method_pdg: typing.Optional[typing.Any] = None


raw_data_default_file_names = RawExtractedDataFiles(
        logging_call_hash='hashes.txt',
        logging_call='logging_calls_json.txt',
        method='method.txt',
        method_ast='method_ast.txt',
        method_pdg='method_pdg.txt')
raw_data_filepath_arg_namespace_field = RawExtractedDataFiles(
    logging_call_hash='logging_call_hashes_filepath',
    logging_call='logging_call_jsons_filepath',
    method='method_jsons_filepath',
    method_ast='method_ast_jsons_filepath',
    method_pdg='method_pdg_jsons_filepath')


def iter_raw_extracted_examples(
        raw_extracted_data_dir: typing.Optional[str] = None, logging_call_hashes_filepath: typing.Optional[str] = None,
        logging_call_jsons_filepath: typing.Optional[str] = None, method_jsons_filepath: typing.Optional[str] = None,
        method_ast_jsons_filepath: typing.Optional[str] = None, method_pdg_jsons_filepath: typing.Optional[str] = None,
        args: typing.Optional[argparse.Namespace] = None, show_progress_bar: bool = False,
        sample_rate: typing.Optional[float] = None, verify_aligned: bool = True, require_logging_call: bool = False,
        require_logging_call_hash: bool = False, require_method: bool = False, require_method_ast: bool = False,
        require_method_pdg: bool = False) -> typing.Iterable[RawExtractedExample]:
    paths_kwargs = argparse.Namespace(
        raw_extracted_data_dir=raw_extracted_data_dir,
        logging_call_hashes_filepath=logging_call_hashes_filepath,
        logging_call_jsons_filepath=logging_call_jsons_filepath,
        method_jsons_filepath=method_jsons_filepath,
        method_ast_jsons_filepath=method_ast_jsons_filepath,
        method_pdg_jsons_filepath=method_pdg_jsons_filepath)
    paths_kwargs = argparse.Namespace(**{k: v for k, v in vars(paths_kwargs).items() if v is not None})
    if len(vars(paths_kwargs)) > 0 and args is not None:
        raise ValueError(f'If `args` namespace is specified, paths cannot be specified as arguments.')
    if args is None:
        args = paths_kwargs

    raw_data_file_paths = RawExtractedDataFiles()
    for field in dataclasses.fields(RawExtractedDataFiles):
        filepath_arg_name = getattr(raw_data_filepath_arg_namespace_field, field.name)
        filepath = None
        if hasattr(args, filepath_arg_name):
            filepath = getattr(args, filepath_arg_name)
            if filepath is not None and not os.path.isfile(filepath):
                raise ValueError(f'File `{filepath}` not exists. Given as path for `{field.name}`.')
        if filepath is None and args.raw_extracted_data_dir:
            filepath = os.path.join(args.raw_extracted_data_dir, getattr(raw_data_default_file_names, field.name))
            if not os.path.isfile(filepath):
                filepath = None
        if filepath is not None:
            setattr(raw_data_file_paths, field.name, filepath)

    if all(val is None for val in dataclasses.asdict(raw_data_file_paths).values()):
        raise ValueError(
            f'Not given paths to any of { {field.name for field in dataclasses.fields(raw_data_file_paths)} }.')

    require = RawExtractedDataFiles(
        logging_call_hash=require_logging_call_hash,
        logging_call=require_logging_call,
        method=require_method,
        method_ast=require_method_ast,
        method_pdg=require_method_pdg)
    for field_name, do_require in dataclasses.asdict(require).items():
        if do_require and getattr(raw_data_file_paths, field_name) is None:
            raise ValueError(f'`{field_name}` is required but not specified.')

    non_null_field_names = tuple(field.name for field in dataclasses.fields(raw_data_file_paths)
                                 if getattr(raw_data_file_paths, field.name) is not None)

    if verify_aligned:
        line_counts_per_file = RawExtractedDataFiles(
            **{field_name: count_lines_in_file(getattr(raw_data_file_paths, field_name))
               for field_name in non_null_field_names})
        all_line_counts = tuple(
            getattr(line_counts_per_file, field.name)
            for field in dataclasses.fields(line_counts_per_file)
            if getattr(line_counts_per_file, field.name) is not None)
        nr_examples = all_line_counts[0]
        if not all(line_count == nr_examples for line_count in all_line_counts):
            lengths = ', '.join(
                f'#{field_name}: {getattr(line_counts_per_file, field_name)}'
                for field_name in non_null_field_names)
            warn(f'Non-equal number of lines in input files. Files are probably not aligned. [{lengths}]')
    else:
        nr_examples = count_lines_in_file(getattr(raw_data_file_paths, non_null_field_names[0]))

    raw_data_files = RawExtractedDataFiles(**{
        field_name: open(getattr(raw_data_file_paths, field_name), 'r', encoding="utf8")
        for field_name in non_null_field_names
    })

    if show_progress_bar:
        progress_bar = iter(tqdm(range(nr_examples)))

    for row_cols_as_tuple in zip(*(getattr(raw_data_files, field_name) for field_name in non_null_field_names)):
        if show_progress_bar:
            next(progress_bar)

        if sample_rate is not None:
            if random.random() > sample_rate:
                continue

        row_cols_as_tuple = tuple(value.strip() for value in row_cols_as_tuple)
        lines = RawExtractedDataFiles(**dict(zip(non_null_field_names, row_cols_as_tuple)))

        example = RawExtractedExample()

        if lines.logging_call_hash is not None:
            example.logging_call_hash = lines.logging_call_hash

        if lines.logging_call is not None:
            logging_call_dict = json.loads(lines.logging_call)
            example.logging_call = SerLoggingCall.from_dict(logging_call_dict)

        if lines.method is not None:
            method_json_dict = json.loads(lines.method)
            example.method = SerMethod.from_dict(method_json_dict)

        if lines.method_ast is not None:
            ast_json_dict = json.loads(lines.method_ast)
            example.method_ast = SerMethodAST.from_dict(ast_json_dict)

        if lines.method_pdg is not None:
            pdg_json_dict = json.loads(lines.method_pdg)
            example.method_pdg = SerMethodPDG.from_dict(pdg_json_dict)

        yield example

    for field_name in non_null_field_names:
        file = getattr(raw_data_files, field_name)
        if file is not None:
            file.close()


def count_lines_in_file(file_path: str):
    with open(file_path, 'rb') as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        return sum(buf.count(b'\n') for buf in bufgen)

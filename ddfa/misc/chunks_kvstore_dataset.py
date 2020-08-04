import os
import io
import torch
import shelve
import itertools
import numpy as np
from warnings import warn
from typing import Optional
from torch.utils.data.dataset import Dataset

from ddfa.dataset_properties import DataFold


__all__ = ['ChunksKVStoreDatasetWriter', 'ChunksKVStoresDataset']


class ChunksKVStoreDatasetWriter:
    KB_IN_BYTES = 1024
    MB_IN_BYTES = 1024 * 1024
    GB_IN_BYTES = 1024 * 1024 * 1024

    def __init__(self, pp_data_path: str, datafold: DataFold, max_chunk_size_in_bytes: int = GB_IN_BYTES):
        self.pp_data_path: str = pp_data_path
        self.datafold: DataFold = datafold
        self.max_chunk_size_in_bytes: int = max_chunk_size_in_bytes
        self.next_example_idx: int = 0
        self.cur_chunk_idx: Optional[int] = None
        self.cur_chunk_size_in_bytes: Optional[int] = None
        self.cur_chunk_nr_examples: Optional[int] = None
        self.cur_chunk_file: Optional[shelve.Shelf] = None
        self.cur_chunk_filepath: Optional[str] = None

    @property
    def total_nr_examples(self) -> int:
        return self.next_example_idx

    def write_example(self, example):
        with io.BytesIO() as bytes_io_stream:
            torch.save(example, bytes_io_stream)
            bytes_io_stream.seek(0)
            binary_serialized_example = bytes_io_stream.read()
            # now i can store `binary_serialized_example` however i want
        example_size_in_bytes = len(binary_serialized_example)
        chunk_file = self.get_cur_chunk_to_write_example_into(example_size_in_bytes)
        chunk_file[str(self.next_example_idx)] = binary_serialized_example
        self.next_example_idx += 1
        self.cur_chunk_nr_examples += 1
        self.cur_chunk_size_in_bytes += example_size_in_bytes
        assert self.cur_chunk_size_in_bytes <= self.max_chunk_size_in_bytes

    def get_cur_chunk_to_write_example_into(self, example_size_in_bytes: int) -> shelve.Shelf:
        assert example_size_in_bytes < self.max_chunk_size_in_bytes
        if self.cur_chunk_file is None or self.cur_chunk_size_in_bytes + example_size_in_bytes >= self.max_chunk_size_in_bytes:
            if self.cur_chunk_idx is None:
                self.cur_chunk_idx = 0
            else:
                self.cur_chunk_idx += 1
                self.close_last_written_chunk()
            self.cur_chunk_filepath = self._get_chunk_filepath(self.cur_chunk_idx)
            if os.path.isfile(self.cur_chunk_filepath):
                if self.cur_chunk_idx == 0:
                    raise ValueError(f'Preprocessed file `{self.cur_chunk_filepath}` already exists. '
                                     f'Please choose another `--pp-data` path or manually delete it.')
                else:
                    warn(f'Overwriting existing preprocessed file `{self.cur_chunk_filepath}`.')
                    os.remove(self.cur_chunk_filepath)
            self.cur_chunk_file = shelve.open(self.cur_chunk_filepath, 'c')
            self.cur_chunk_size_in_bytes = 0
            self.cur_chunk_nr_examples = 0
        return self.cur_chunk_file

    def close_last_written_chunk(self):
        assert self.cur_chunk_nr_examples > 0
        self.cur_chunk_file['len'] = self.cur_chunk_nr_examples
        self.cur_chunk_file.close()
        self.cur_chunk_file = None

    def _get_chunk_filepath(self, chunk_idx: int) -> str:
        return os.path.join(self.pp_data_path, f'pp_{self.datafold.value.lower()}.{chunk_idx}.pt')

    def enforce_no_further_chunks(self):
        # Remove old extra file chunks
        for chunk_idx_to_remove in itertools.count(start=self.cur_chunk_idx + 1):
            chunk_filepath = self._get_chunk_filepath(chunk_idx_to_remove)
            if not os.path.isfile(chunk_filepath):
                break
            warn(f'Removing existing preprocessed file `{chunk_filepath}`.')
            os.remove(chunk_filepath)


class ChunksKVStoresDataset(Dataset):
    def __init__(self, datafold: DataFold, pp_data_path: str):
        self.datafold = datafold
        self.pp_data_path = pp_data_path
        self._pp_data_chunks_filepaths = []
        self._kvstore_chunks = []
        self._kvstore_chunks_lengths = []
        for chunk_idx in itertools.count():
            filepath = os.path.join(self.pp_data_path, f'pp_{self.datafold.value.lower()}.{chunk_idx}.pt')
            if not os.path.isfile(filepath) and not os.path.isfile(filepath + '.dat'):
                if chunk_idx == 0:
                    raise ValueError(f'Not found `{self.datafold}` dataset in path `{self.pp_data_path}`.')
                else:
                    break
            self._pp_data_chunks_filepaths.append(filepath)
            kvstore = shelve.open(filepath, 'r')
            self._kvstore_chunks.append(kvstore)
            self._kvstore_chunks_lengths.append(kvstore['len'])
        self._len = sum(self._kvstore_chunks_lengths)
        self._kvstore_chunks_lengths = np.array(self._kvstore_chunks_lengths)
        self._kvstore_chunks_stop_indices = np.cumsum(self._kvstore_chunks_lengths)
        self._kvstore_chunks_start_indices = self._kvstore_chunks_stop_indices - self._kvstore_chunks_lengths
        # TODO: add hash of task props & model HPs to perprocessed file name.

    def _get_chunk_idx_contains_item(self, item_idx: int) -> int:
        assert item_idx < self._len
        cond = (self._kvstore_chunks_start_indices <= item_idx) & (self._kvstore_chunks_stop_indices > item_idx)
        assert np.sum(cond) == 1
        found_idx = np.where(cond)
        assert len(found_idx) == 1
        return int(found_idx[0])

    def __len__(self):
        return self._len

    def __del__(self):
        for chunk_kvstore in self._kvstore_chunks:
            chunk_kvstore.close()

    def __getitem__(self, idx):
        assert isinstance(idx, int) and idx <= self._len
        chunk_kvstore = self._kvstore_chunks[self._get_chunk_idx_contains_item(idx)]
        binary_serialized_example = chunk_kvstore[str(idx)]
        with io.BytesIO(binary_serialized_example) as bytes_io_stream:
            bytes_io_stream.seek(0)
            return torch.load(bytes_io_stream)

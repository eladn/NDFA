import os
import io
import abc
import math
import torch
import shutil
import itertools
import numpy as np
from warnings import warn
from typing import Optional, Tuple, Sequence
from torch.utils.data.dataset import Dataset


__all__ = ['ChunkedRandomAccessDatasetWriter', 'ChunkedRandomAccessDataset']


def prettyprint_filesize(size_bytes: int):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


class KeyValueStoreInterface(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def open(cls, path, mode, compression) -> 'KeyValueStoreInterface':
        ...

    @abc.abstractmethod
    def get_value_by_key(self, key: str) -> bytes:
        ...

    @abc.abstractmethod
    def write_member(self, key: str, value: bytes) -> int:
        ...

    @abc.abstractmethod
    def close(self):
        ...

    @classmethod
    @abc.abstractmethod
    def get_new_and_truncate_write_mode(cls) -> 'str':
        ...

    @classmethod
    @abc.abstractmethod
    def get_read_mode(cls) -> 'str':
        ...

    def get_values_by_keys(self, keys: Sequence[str]) -> Tuple[bytes, ...]:
        return tuple(self.get_value_by_key(key) for key in keys)


class DBMKeyValueStore(KeyValueStoreInterface):
    def __init__(self, path, mode):
        from dbm import open as dbm_open
        self.dbm = dbm_open(path, mode)

    @classmethod
    def open(cls, path, mode, compression) -> 'DBMKeyValueStore':
        assert compression == 'none'
        return DBMKeyValueStore(path=path, mode=mode)

    def close(self):
        self.dbm.close()

    def get_value_by_key(self, key: str) -> bytes:
        return self.dbm[key.encode('ascii')]

    def write_member(self, key: str, value: bytes) -> int:
        self.dbm[key.encode('ascii')] = value
        return len(value)

    @classmethod
    def get_new_and_truncate_write_mode(cls) -> 'str':
        return 'n'

    @classmethod
    def get_read_mode(cls) -> 'str':
        return 'r'


class ZipKeyValueStore(KeyValueStoreInterface):
    def __init__(self, path, mode, compression):
        from zipfile import ZipFile, ZIP_LZMA, ZIP_BZIP2, ZIP_STORED, ZIP_DEFLATED
        compression_translation = {'none': ZIP_STORED, 'gzip': ZIP_DEFLATED, 'bz2': ZIP_BZIP2, 'lzma': ZIP_LZMA}
        if compression not in compression_translation:
            raise ValueError(f'Unsupported compression type `{compression}` for zip storage method. '
                             f'Supported methods: {list(compression_translation.keys())}.')
        compression = compression_translation[compression]
        self.zip_file = ZipFile(path, mode, compression=compression)

    @classmethod
    def open(cls, path, mode, compression) -> 'ZipKeyValueStore':
        return ZipKeyValueStore(path=path, mode=mode, compression=compression)

    def close(self):
        self.zip_file.close()

    def get_value_by_key(self, key: str) -> bytes:
        self.zip_file.getinfo(key)  # verify it is indeed here
        return self.zip_file.read(key)

    def write_member(self, key: str, value: bytes) -> int:
        # with tempfile.NamedTemporaryFile(mode='wb') as tmp_file:
        #     tmp_file.write(value)
        #     tmp_file.seek(0)
        #     self.zip_file.write(tmp_file.name, arcname=key)
        self.zip_file.writestr(key, value)
        zip_info = self.zip_file.getinfo(key)
        return zip_info.compress_size

    @classmethod
    def get_new_and_truncate_write_mode(cls) -> 'str':
        return 'x'

    @classmethod
    def get_read_mode(cls) -> 'str':
        return 'r'


class TarKeyValueStore(KeyValueStoreInterface):
    def __init__(self, path, mode, compression):
        from tarfile import open as tarfile_open
        compression_translation = {'none': '', 'gzip': 'gz', 'bz2': 'bz2', 'lzma': 'xz'}
        if compression not in compression_translation:
            raise ValueError(f'Unsupported compression type `{compression}` for tar storage method. '
                             f'Supported methods: {list(compression_translation.keys())}.')
        compression_suffix = compression_translation[compression]
        self.tar_file = tarfile_open(path, f'{mode}:{compression_suffix}')

    @classmethod
    def open(cls, path, mode, compression) -> 'TarKeyValueStore':
        return TarKeyValueStore(path=path, mode=mode, compression=compression)

    def close(self):
        self.tar_file.close()

    def get_value_by_key(self, key: str) -> bytes:
        with self.tar_file.extractfile(key) as file_stream:
            file_stream.seek(0)
            return file_stream.read()

    def write_member(self, key: str, value: bytes) -> int:
        from tarfile import TarInfo
        with io.BytesIO(value) as value_as_bytes_stream:
            value_as_bytes_stream.seek(0)
            tar_info = TarInfo(name=key)
            tar_info.size = len(value)
            # tar_info = self.tar_file.gettarinfo(arcname=key, fileobj=value_as_bytes_stream)
            value_as_bytes_stream.seek(0)
            self.tar_file.addfile(tar_info, value_as_bytes_stream)
        tar_info = self.tar_file.getmember(key)
        return tar_info.size  # it is not the compressed size :(

    @classmethod
    def get_new_and_truncate_write_mode(cls) -> 'str':
        return 'x'

    @classmethod
    def get_read_mode(cls) -> 'str':
        return 'r'


class RocksDBKeyValueStore(KeyValueStoreInterface):
    def __init__(self, path, mode, compression):
        compressions_translation = {'bz2': 'bzip2', 'none': 'no', 'lzma': 'lz4', 'gzip': 'zlib'}
        if compression in compressions_translation:
            compression = compressions_translation[compression]
        allowed_compressions = \
            {'no', 'snappy', 'zlib', 'bzip2', 'lz4', 'lz4hc', 'xpress', 'zstd', 'zstdnotfinal', 'disable'}
        if compression not in allowed_compressions:
            raise ValueError(f'Unsupported compression type `{compression}` for RocksDB storage method. '
                             f'Supported methods: {list(allowed_compressions)}.')
        import rocksdb
        compression_type = getattr(rocksdb.CompressionType, f'{compression}_compression')
        if mode == 'x' and os.path.exists(path):
            shutil.rmtree(path)  # TODO: fix it to remove all chunks!
        # Note: These are pre-coocked production-ready options;
        #       Read more here: https://python-rocksdb.readthedocs.io/en/latest/tutorial/index.html#open
        options = rocksdb.Options(
            create_if_missing=mode == 'x',
            compression=compression_type,
            max_open_files=300000,
            write_buffer_size=67108864,
            max_write_buffer_number=3,
            target_file_size_base=67108864,
            table_factory=rocksdb.BlockBasedTableFactory(
                filter_policy=rocksdb.BloomFilterPolicy(10),
                block_cache=rocksdb.LRUCache(2 * (1024 ** 3)),
                block_cache_compressed=rocksdb.LRUCache(500 * (1024 ** 2))))
        self.db = rocksdb.DB(path, options)

    @classmethod
    def open(cls, path, mode, compression) -> 'RocksDBKeyValueStore':
        return RocksDBKeyValueStore(path=path, mode=mode, compression=compression)

    def close(self):
        del self.db

    def get_value_by_key(self, key: str) -> bytes:
        return self.db.get(key.encode('ascii'))

    def get_values_by_keys(self, keys: Sequence[str]) -> Tuple[bytes, ...]:
        encoded_keys = [key.encode('ascii') for key in keys]
        value_by_key = self.db.multi_get(encoded_keys)
        return tuple(value_by_key[key] for key in encoded_keys)

    def write_member(self, key: str, value: bytes) -> int:
        self.db.put(key.encode('ascii'), value)
        # return len(value)  # it is not the actual total & compressed size :( I don't know how to get it.
        return 0  # so chunks won't be created.

    @classmethod
    def get_new_and_truncate_write_mode(cls) -> 'str':
        return 'x'

    @classmethod
    def get_read_mode(cls) -> 'str':
        return 'r'


def get_key_value_store(storage_method: str = 'dbm', compression_method: str = 'none'):
    kv_store_methods_by_name = {
        'dbm': DBMKeyValueStore,
        'zip': ZipKeyValueStore,
        'tar': TarKeyValueStore,
        'rocksdb': RocksDBKeyValueStore}
    return kv_store_methods_by_name[storage_method]


class ChunkedRandomAccessDatasetWriter:
    KB_IN_BYTES = 1024
    MB_IN_BYTES = 1024 * 1024
    GB_IN_BYTES = 1024 * 1024 * 1024

    def __init__(self, pp_data_path_prefix: str,
                 max_chunk_size_in_bytes: int = GB_IN_BYTES,
                 override: bool = False,
                 storage_method: str = 'dbm',
                 compression_method: str = 'none'):
        self.pp_data_path_prefix: str = pp_data_path_prefix
        self.max_chunk_size_in_bytes: int = max_chunk_size_in_bytes
        self.override: bool = override
        self.next_example_idx: int = 0
        self.cur_chunk_idx: Optional[int] = None
        self.cur_chunk_size_in_bytes: Optional[int] = None
        self.cur_chunk_nr_examples: Optional[int] = None
        self.cur_chunk_file: Optional[KeyValueStoreInterface] = None
        self.cur_chunk_filepath: Optional[str] = None
        self.key_value_store_type = get_key_value_store(
            storage_method=storage_method, compression_method=compression_method)
        self.compression_method = compression_method

    @property
    def total_nr_examples(self) -> int:
        return self.next_example_idx

    def write_example(self, example):
        if isinstance(example, bytes):
            binary_serialized_example = example
        else:
            with io.BytesIO() as bytes_io_stream:
                torch.save(example, bytes_io_stream)
                binary_serialized_example = bytes_io_stream.getvalue()
        # now the example is already serialized (into bytes) and we directly export it as-is to `dbm` KV-store.
        avg_example_size = self.cur_chunk_size_in_bytes / self.cur_chunk_nr_examples \
            if self.cur_chunk_nr_examples is not None and self.cur_chunk_nr_examples > 0 else 0
        chunk_file = self.get_cur_chunk_to_write_example_into(avg_example_size)
        example_size_in_bytes = chunk_file.write_member(str(self.next_example_idx), binary_serialized_example)
        self.next_example_idx += 1
        self.cur_chunk_nr_examples += 1
        self.cur_chunk_size_in_bytes += example_size_in_bytes
        assert self.cur_chunk_size_in_bytes <= \
               self.max_chunk_size_in_bytes + max(0, example_size_in_bytes - avg_example_size)

    def get_cur_chunk_to_write_example_into(self, example_size_in_bytes: int = 0) \
            -> KeyValueStoreInterface:
        assert example_size_in_bytes < self.max_chunk_size_in_bytes
        if self.cur_chunk_file is None or \
                self.cur_chunk_size_in_bytes + example_size_in_bytes >= self.max_chunk_size_in_bytes:
            if self.cur_chunk_idx is None:
                self.cur_chunk_idx = 0
            else:
                self.close_last_written_chunk()
                self.cur_chunk_idx += 1
            self.cur_chunk_filepath = self._get_chunk_filepath(self.cur_chunk_idx)
            cur_chunk_files_found_in_pp_dir = [
                filename for filename in os.listdir(os.path.dirname(self.cur_chunk_filepath))
                if filename.startswith(os.path.basename(self.cur_chunk_filepath))]
            if len(cur_chunk_files_found_in_pp_dir) > 0:
                if self.cur_chunk_idx == 0 and not self.override:
                    raise ValueError(
                        f'Preprocessed files `{cur_chunk_files_found_in_pp_dir}` already exists '
                        f'in dir `{os.path.dirname(self.cur_chunk_filepath)}`. '
                        f'Please either specify `--pp-override` argument, choose another `--pp-data`, '
                        f'or manually delete it.')
                else:
                    warn(f'Overwriting existing preprocessed files {cur_chunk_files_found_in_pp_dir} '
                         f'in dir `{os.path.dirname(self.cur_chunk_filepath)}`.')
                    for filename in cur_chunk_files_found_in_pp_dir:
                        shutil.rmtree(os.path.join(os.path.dirname(self.cur_chunk_filepath), filename))
            self.cur_chunk_file = self.key_value_store_type.open(
                self.cur_chunk_filepath, self.key_value_store_type.get_new_and_truncate_write_mode(),
                compression=self.compression_method)
            self.cur_chunk_size_in_bytes = 0
            self.cur_chunk_nr_examples = 0
        return self.cur_chunk_file

    def close_last_written_chunk(self):
        assert self.cur_chunk_nr_examples > 0
        print(f'Closing chunk #{self.cur_chunk_idx} with {self.cur_chunk_nr_examples:,} examples '
              f'and total size of {prettyprint_filesize(self.cur_chunk_size_in_bytes)}.')
        self.cur_chunk_file.write_member('len', int(self.cur_chunk_nr_examples).to_bytes(8, 'little'))
        self.cur_chunk_file.close()
        self.cur_chunk_file = None

    def _get_chunk_filepath(self, chunk_idx: int) -> str:
        return f'{self.pp_data_path_prefix}.{chunk_idx}.pt'

    def enforce_no_further_chunks(self):
        # Remove old extra file chunks
        for chunk_idx_to_remove in itertools.count(start=self.cur_chunk_idx + 1):
            chunk_filepath = self._get_chunk_filepath(chunk_idx_to_remove)
            chunk_files_found_in_pp_dir = [
                filename for filename in os.listdir(os.path.dirname(chunk_filepath))
                if filename.startswith(os.path.basename(chunk_filepath))]
            if len(chunk_files_found_in_pp_dir) == 0:
                break
            warn(f'Removing existing preprocessed files {chunk_files_found_in_pp_dir} '
                 f'in dir `{os.path.dirname(chunk_filepath)}`.')
            for filename in chunk_files_found_in_pp_dir:
                shutil.rmtree(os.path.join(os.path.dirname(chunk_filepath), filename))


class ChunkedRandomAccessDataset(Dataset):
    def __init__(self, pp_data_path_prefix: str,
                 storage_method: str = 'dbm',
                 compression_method: str = 'none'):
        self.pp_data_path_prefix = pp_data_path_prefix
        self.key_value_store_type = get_key_value_store(
            storage_method=storage_method, compression_method=compression_method)
        self.compression_method = compression_method
        self._pp_data_chunks_filepaths = []
        self._kvstore_opened_chunks_per_pid = {}
        self._kvstore_chunks_lengths = []
        for chunk_idx in itertools.count():
            filepath = f'{self.pp_data_path_prefix}.{chunk_idx}.pt'
            if not os.path.exists(filepath) and not os.path.exists(filepath + '.dat'):
                if chunk_idx == 0:
                    raise ValueError(f'Could not find dataset in path `{self.pp_data_path_prefix}`.')
                else:
                    break
            self._pp_data_chunks_filepaths.append(filepath)
            kvstore = self._open_chunk(chunk_idx=chunk_idx)
            self._kvstore_chunks_lengths.append(int.from_bytes(kvstore.get_value_by_key('len'), 'little'))
            kvstore.close()
        self._len = sum(self._kvstore_chunks_lengths)
        print(f'Loaded dataset `{os.path.basename(pp_data_path_prefix)}` of {self._len:,} examples '
              f'over {len(self._kvstore_chunks_lengths)} chunk{"" if len(self._kvstore_chunks_lengths) == 1 else "s"}.')
        self._kvstore_chunks_lengths = np.array(self._kvstore_chunks_lengths)
        self._kvstore_chunks_stop_indices = np.cumsum(self._kvstore_chunks_lengths)
        self._kvstore_chunks_start_indices = self._kvstore_chunks_stop_indices - self._kvstore_chunks_lengths
        self.parallel_deserialize_worker = None

    def _open_chunk(self, chunk_idx: int) -> KeyValueStoreInterface:
        return self.key_value_store_type.open(
            path=self._pp_data_chunks_filepaths[chunk_idx],
            mode=self.key_value_store_type.get_read_mode(),
            compression=self.compression_method)

    def _get_chunk(self, chunk_idx: int):
        if os.getpid() not in self._kvstore_opened_chunks_per_pid:
            self._kvstore_opened_chunks_per_pid[os.getpid()] = {}
        opened_chunks = self._kvstore_opened_chunks_per_pid[os.getpid()]
        if chunk_idx not in opened_chunks:
            opened_chunks[chunk_idx] = self._open_chunk(chunk_idx=chunk_idx)
        return opened_chunks[chunk_idx]

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
        if hasattr(self, '_kvstore_opened_chunks_per_pid'):
            for kvstore_opened_chunks in self._kvstore_opened_chunks_per_pid.values():
                for chunk_kvstore in kvstore_opened_chunks.values():
                    chunk_kvstore.close()
        if self.parallel_deserialize_worker is not None:
            del self.parallel_deserialize_worker

    def __getitem__(self, possibly_batched_index):
        if isinstance(possibly_batched_index, int):
            assert possibly_batched_index <= self._len
            chunk_kvstore = self._get_chunk(chunk_idx=self._get_chunk_idx_contains_item(possibly_batched_index))
            binary_serialized_example = chunk_kvstore.get_value_by_key(str(possibly_batched_index))
            with io.BytesIO(binary_serialized_example) as bytes_io_stream:
                bytes_io_stream.seek(0)
                return torch.load(bytes_io_stream)
        elif isinstance(possibly_batched_index, (list, tuple)):
            assert all(isinstance(index, int) and index <= self._len for index in possibly_batched_index)
            assert len(self._kvstore_chunks_lengths) == 1
            chunk_kvstore = self._get_chunk(chunk_idx=self._get_chunk_idx_contains_item(possibly_batched_index[0]))
            binary_serialized_examples = chunk_kvstore.get_values_by_keys([str(idx) for idx in possibly_batched_index])

            # concurrent version:
            #   [doesn't help. i checked on my 2 cores Mac with batch-size=32:
            #    for 2 "jobs" it adds ~10sec, for 4 "jobs" it adds even more]
            # from joblib import Parallel, delayed
            # if self.parallel_deserialize_worker is None:
            #     self.parallel_deserialize_worker = Parallel(n_jobs=2)
            # def load_example(binary_serialized_example: bytes):
            #     with io.BytesIO(binary_serialized_example) as bytes_io_stream:
            #         bytes_io_stream.seek(0)
            #         return torch.load(bytes_io_stream)
            # loaded_examples = self.parallel_deserialize_worker(
            #     delayed(load_example)(binary_serialized_example)
            #     for binary_serialized_example in binary_serialized_examples)
            # return loaded_examples

            loaded_examples = []
            for binary_serialized_example in binary_serialized_examples:
                with io.BytesIO(binary_serialized_example) as bytes_io_stream:
                    bytes_io_stream.seek(0)
                    loaded_examples.append(torch.load(bytes_io_stream))
            return loaded_examples
        else:
            assert False

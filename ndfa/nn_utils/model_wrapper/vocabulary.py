import os
import torch
import itertools
import pickle as pkl
from typing import Optional, Callable, Iterable, Collection, List, Union
from collections import Counter


__all__ = ['Vocabulary']


# TODO: create VocabProperties (confclass) which has `min_word_freq`, `max_vocab_size`


class Vocabulary:
    def __init__(self, name: str, all_words_sorted_by_idx: List[str],
                 params: Optional[dict] = None,
                 special_words_sorted_by_idx: Collection[str] = ()):
        self.name = name
        self.special_words = tuple(special_words_sorted_by_idx)
        self.idx2word = self.special_words + tuple(all_words_sorted_by_idx)
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.params = {} if params is None else params

    def __len__(self):
        return len(self.idx2word)

    def size_wo_specials(self) -> int:
        return len(self.idx2word) - len(self.special_words)

    def get_word_idx(self, word: str) -> Union[int, torch.Tensor]:
        assert word in self.word2idx
        return self.word2idx[word]

    def get_word_idx_or_unk(self, word: str, unk_word: str = '<UNK>',
                            as_tensor: bool = False) -> Union[int, torch.Tensor]:
        idx = self.word2idx[word] if word in self.word2idx else self.word2idx[unk_word]
        return torch.tensor([idx]) if as_tensor else idx

    @staticmethod
    def load_or_create(
            preprocessed_data_dir_path: str, vocab_name: str, special_words_sorted_by_idx: Collection[str] = (),
            min_word_freq: Optional[int] = None, max_vocab_size_wo_specials: Optional[int] = None,
            carpus_generator: Optional[Callable[[], Iterable[str]]] = None) -> 'Vocabulary':
        # TODO: get a `VocabProperties` confclass instead of params `min_word_freq`, `max_vocab_size`.
        vocabulary_params = dict(
            min_word_freq=min_word_freq,
            max_vocab_size_wo_specials=max_vocab_size_wo_specials,
            nr_special_words=len(special_words_sorted_by_idx))
        vocabulary_params_as_str = '_'.join(
            f'{key}={val}' for key, val in vocabulary_params.items() if val is not None)
        vocabulary_filename = f'vocab_{vocab_name}_{vocabulary_params_as_str}.pkl'
        vocabulary_file_path = os.path.join(preprocessed_data_dir_path, vocabulary_filename)
        if os.path.isfile(vocabulary_file_path):
            with open(vocabulary_file_path, 'br') as vocabulary_file:
                print(f'Loading vocabulary `{vocab_name}` from file @ `{vocabulary_file_path}`.')
                all_vocab_words_sorted_by_idx = pkl.load(vocabulary_file)
                assert isinstance(all_vocab_words_sorted_by_idx, list)
                assert all(isinstance(word, str) for word in all_vocab_words_sorted_by_idx)
        else:
            all_carpus_words_with_freqs = Vocabulary.load_or_create_carpus_words_freqs(
                preprocessed_data_dir_path=preprocessed_data_dir_path, vocab_name=vocab_name,
                carpus_generator=carpus_generator)
            print(f'Creating vocabulary `{vocab_name}` ..')
            carpus_iterator_wo_low_freq = (
                word for word, freq in all_carpus_words_with_freqs.items()
                if (min_word_freq is None or freq >= min_word_freq))
            all_vocab_words_sorted_by_idx = list(
                carpus_iterator_wo_low_freq if max_vocab_size_wo_specials is None else
                itertools.islice(carpus_iterator_wo_low_freq, max_vocab_size_wo_specials))
            with open(vocabulary_file_path, 'bw') as vocabulary_file:
                pkl.dump(all_vocab_words_sorted_by_idx, vocabulary_file)
            print(f'Done creating vocabulary `{vocab_name}` [stored @ `{vocabulary_file_path}`].')
        return Vocabulary(
            name=vocab_name, all_words_sorted_by_idx=all_vocab_words_sorted_by_idx, params=vocabulary_params,
            special_words_sorted_by_idx=special_words_sorted_by_idx)

    @staticmethod
    def load_or_create_carpus_words_freqs(
            preprocessed_data_dir_path: str, vocab_name: str,
            carpus_generator: Optional[Callable[[], Iterable[str]]] = None) -> Counter:
        # TODO: sort words lexicography as a secondary order if both have the same freq.
        carpus_word_freqs_filename = f'carpus_words_freq_{vocab_name}.pkl'
        carpus_word_freqs_file_path = os.path.join(preprocessed_data_dir_path, carpus_word_freqs_filename)
        if os.path.isfile(carpus_word_freqs_file_path):
            with open(carpus_word_freqs_file_path, 'br') as carpus_word_freqs_file:
                print(f'Loading carpus frequencies of vocabulary `{vocab_name}` from file @ `{carpus_word_freqs_file_path}`.')
                all_carpus_words_with_freqs = pkl.load(carpus_word_freqs_file)
                assert isinstance(all_carpus_words_with_freqs, Counter)
                assert all(isinstance(word, str) for word in all_carpus_words_with_freqs.keys())
        elif carpus_generator is not None:
            print(f'Creating carpus frequencies of vocabulary `{vocab_name}` ..')
            all_carpus_words_with_freqs = Counter(iter(carpus_generator()))
            with open(carpus_word_freqs_file_path, 'bw') as carpus_word_freqs_file:
                pkl.dump(all_carpus_words_with_freqs, carpus_word_freqs_file)
            print(f'Done creating carpus frequencies of vocabulary `{vocab_name}` [stored @ `{carpus_word_freqs_file_path}`].')
        else:
            raise ValueError(
                f'Error while trying to load or create a vocabulary ({vocab_name}): '
                f'Neither a stored vocabulary found nor a stored carpus words frequencies found nor '
                f'a carpus generator is supplied (have you supplied the raw training dataset?).')
        return all_carpus_words_with_freqs




from typing import Optional
from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field


__all__ = ['EmbeddingWithUnknownsParams']


@dataclass
class EmbeddingWithUnknownsParams:
    obfuscation_type: str = conf_field(
        default='replace_oov_and_random',
        choices=('none', 'add_all', 'replace_all', 'replace_oovs',
                 'replace_random', 'replace_oov_and_random'))
    replace_random_in_inference: bool = conf_field(
        default=False)
    obfuscation_embeddings_type: str = conf_field(
        default='learnable',
        choices=('learnable', 'fixed_orthogonal'))
    obfuscation_rate: float = conf_field(
        default=0.3)
    use_vocab: bool = conf_field(default=True)
    use_hashing_trick: bool = conf_field(default=False)
    nr_obfuscation_words: Optional[int] = conf_field(default=64)

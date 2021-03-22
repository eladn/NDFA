from dataclasses import dataclass
from confclass import confparam


__all__ = ['EmbeddingWithUnknownsParams']


@dataclass
class EmbeddingWithUnknownsParams:
    obfuscation_type: str = confparam(
        default='replace_oov_and_random',
        choices=('none', 'add_all', 'replace_all', 'replace_oovs',
                 'replace_random', 'replace_oov_and_random'))
    obfuscation_embeddings_type: str = confparam(
        default='learnable',
        choices=('learnable', 'fixed_orthogonal'))
    obfuscation_rate: float = confparam(
        default=0.3)
    use_vocab: bool = confparam(default=True)
    use_hashing_trick: bool = confparam(default=False)

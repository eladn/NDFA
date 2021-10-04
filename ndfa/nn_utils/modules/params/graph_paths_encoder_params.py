from enum import Enum


__all__ = ['EdgeTypeInsertionMode']


class EdgeTypeInsertionMode(Enum):
    Without = 'Without'
    AsStandAloneToken = 'AsStandAloneToken'
    MixWithNodeEmbedding = 'MixWithNodeEmbedding'

__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-10-05"

from enum import Enum
from typing import Optional
from dataclasses import dataclass

from ndfa.misc.configurations_utils import HasDispatchableField, DispatchField, conf_field
from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams
from ndfa.nn_utils.modules.params.scatter_combiner_params import ScatterCombinerParams
from ndfa.nn_utils.modules.params.n_grams_params import NGramsParams
from ndfa.nn_utils.modules.params.graph_paths_encoder_params import EdgeTypeInsertionMode
from ndfa.nn_utils.modules.params.sequence_combiner_params import SequenceCombinerParams


__all__ = ['CFGPathsMacroEncoderParams']


@dataclass
class CFGPathsMacroEncoderParams(HasDispatchableField):
    class PathsType(Enum):
        ControlFlow = 'ControlFlow'
        DataDependencyAndControlFlow = 'DataDependencyAndControlFlow'

    class OutputType(Enum):
        FoldNodeOccurrencesToNodeEncodings = 'FoldNodeOccurrencesToNodeEncodings'
        SetOfPaths = 'SetOfPaths'

    output_type: OutputType = conf_field(
        default=OutputType.FoldNodeOccurrencesToNodeEncodings)
    nodes_folding_params: Optional[ScatterCombinerParams] = conf_field(
        default_factory=ScatterCombinerParams)
    paths_combining_params: Optional[SequenceCombinerParams] = conf_field(
        default_factory=SequenceCombinerParams)

    path_sequence_encoder: Optional[SequenceEncoderParams] = conf_field(
        default_factory=SequenceEncoderParams,
        arg_prefix='sequence-encoder')

    edge_types_insertion_mode: Optional[EdgeTypeInsertionMode] = conf_field(
        default=EdgeTypeInsertionMode.AsStandAloneToken)

    is_ngrams: bool = conf_field(
        default=False)
    ngrams: Optional[NGramsParams] = conf_field(
        default_factory=NGramsParams)

    paths_type: PathsType = conf_field(
        default=PathsType.ControlFlow)

    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'output_type', {
                cls.OutputType.FoldNodeOccurrencesToNodeEncodings: ['nodes_folding_params'],
                cls.OutputType.SetOfPaths: ['paths_combining_params']
            }))
        cls.register_dispatch_field(DispatchField(
            'is_ngrams', {
                True: ['ngrams'],
                False: []
            }))

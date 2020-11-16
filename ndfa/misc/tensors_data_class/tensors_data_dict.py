import dataclasses
from typing import List, Tuple, Dict, Generic, TypeVar, final

from .tensors_data_class import TensorsDataClass
from .misc import CollateData


DictKeyT = TypeVar('DictKeyT')
DictValueT = TypeVar('DictValueT')


@final
@dataclasses.dataclass
class TensorsDataDict(TensorsDataClass, Generic[DictKeyT, DictValueT]):
    dict: Dict[DictKeyT, DictValueT] = dataclasses.field(default_factory=dict)

    def access_field(self, name: str):
        return self.dict[name]

    def get_all_traversable_data_field_names(self) -> Tuple[str]:
        return dataclasses.fields(self)

    @classmethod
    def _collate_first_pass(cls, inputs: List['TensorsDataDict'], collate_data: CollateData) -> 'TensorsDataDict':
        assert all(isinstance(inp, TensorsDataDict) for inp in inputs)
        all_keys = {key for dct in inputs for key in dct.keys()}
        batched_obj = TensorsDataDict(dict={
            key: cls.collate_values(
                tuple(dct[key] for dct in inputs if key in dct),
                collate_data=collate_data)
            for key in all_keys})
        batched_obj._batch_size = len(inputs)
        return batched_obj

    def post_collate_indices_fix(
            self, parents: Tuple['TensorsDataClass', ...],
            fields_path: Tuple[str, ...], collate_data: CollateData):
        for key, value in self.dict.items():
            if isinstance(value, TensorsDataClass):
                value.post_collate_indices_fix(
                    parents=parents + (self,), fields_path=fields_path + (key,),
                    collate_data=collate_data)

    def post_collate_remove_unnecessary_collate_info(self):
        for key, value in self.dict.items():
            if isinstance(value, TensorsDataClass):
                value.post_collate_remove_unnecessary_collate_info()

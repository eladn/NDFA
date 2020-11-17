import dataclasses
from typing import List, Tuple, Dict, TypeVar, final, Iterable, Mapping

from .tensors_data_class import TensorsDataClass
from .misc import CollateData, compose_fns


DictKeyT = TypeVar('DictKeyT')
DictValueT = TypeVar('DictValueT')


@final
@dataclasses.dataclass
class TensorsDataDict(TensorsDataClass, Mapping[DictKeyT, DictValueT]):
    _dict: Dict[DictKeyT, DictValueT] = dataclasses.field(default_factory=dict)

    def __init__(self, dct: Dict[DictKeyT, DictValueT] = None, /, **kwargs):
        super(TensorsDataDict, self).__init__()

    def __new__(cls, dct: Dict[DictKeyT, DictValueT] = None, /, **kwargs):
        obj = super(TensorsDataDict, cls).__new__(cls)
        dct = {} if dct is None else dct
        dct = {**dct, **kwargs}
        obj._dict = dct
        return obj

    @classmethod
    def factory(cls, kwargs: Dict) -> 'TensorsDataDict':
        assert {field.name for field in dataclasses.fields(cls) if field.init} == {'_dict'}
        assert '_dict' not in kwargs.keys()
        dataclass_fields_not_to_init = {
            field.name: field for field in dataclasses.fields(cls) if not field.init}
        new_obj = cls({
            key: val for key, val in kwargs.items()
            if key not in dataclass_fields_not_to_init})
        for field_name in set(dataclass_fields_not_to_init.keys()) & set(kwargs.keys()):
            setattr(new_obj, field_name, kwargs[field_name])
        return new_obj

    def get_all_fields(self) -> Tuple[str, ...]:
        return tuple((set(self._dict.keys()) | set(super(TensorsDataDict, self).get_all_fields())) - {'_dict'})

    @classmethod
    def _collate_first_pass(cls, inputs: List['TensorsDataDict'], collate_data: CollateData) -> 'TensorsDataDict':
        assert all(isinstance(inp, TensorsDataDict) for inp in inputs)
        all_keys = {key for dct in inputs for key in dct._dict.keys()}
        batched_obj = TensorsDataDict({
            key: cls.collate_values(
                tuple(dct._dict[key] for dct in inputs if key in dct._dict),
                collate_data=collate_data)
            for key in all_keys})
        batched_obj._batch_size = len(inputs)
        return batched_obj

    # TODO: remove this override - unnecessary!
    # def post_collate_indices_fix(
    #         self, parents: Tuple['TensorsDataClass', ...],
    #         fields_path: Tuple[str, ...], collate_data: CollateData):
    #     for key, value in self._dict.items():
    #         if isinstance(value, TensorsDataClass):
    #             value.post_collate_indices_fix(
    #                 parents=parents + (self,), fields_path=fields_path + (key,),
    #                 collate_data=collate_data)

    # TODO: remove this override - unnecessary!
    # def post_collate_remove_unnecessary_collate_info(self):
    #     for key, value in self._dict.items():
    #         if isinstance(value, TensorsDataClass):
    #             value.post_collate_remove_unnecessary_collate_info()

    def __getitem__(self, item):
        # `__getitem__` is overridden to support `lazy_map()`.
        if hasattr(self, '_lazy_map_fns_per_field') and item in self._lazy_map_fns_per_field:
            old_val = self._dict[item]
            composed_map_fn = compose_fns(*(fn for fn, _ in self._lazy_map_fns_per_field[item]))
            self._dict[item] = composed_map_fn(old_val)
            del self._lazy_map_fns_per_field[item]
            if len(self._lazy_map_fns_per_field) == 0:
                del self._lazy_map_fns_per_field
            self._lazy_map_usage_history.add(item)
        return self._dict[item]

    def access_field(self, name: str):
        if isinstance(name, str) and hasattr(super(TensorsDataDict, self), name):
            return super(TensorsDataDict, self).access_field(name)
        return self[name]

    def access_field_wo_applying_lazy_maps(self, name: str):
        if isinstance(name, str) and hasattr(super(TensorsDataDict, self), name):
            return super(TensorsDataDict, self).access_field_wo_applying_lazy_maps(name)
        return self._dict[name]

    def items(self) -> Iterable[Tuple[DictKeyT, DictValueT]]:
        for key in self._dict.keys():
            yield key, self[key]

    def values(self) -> Iterable[DictValueT]:
        return (self[key] for key in self.keys())

    def keys(self) -> Iterable[DictKeyT]:
        return self._dict.keys()

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return self._dict.keys()

import torch
import dataclasses
from typing import List, Optional, Tuple, Dict, Set

from .misc import compose_fns, CollateData, CollatableValuesTuple, MapFn


__all__ = ['TensorsDataClass']


# TODO:
#  Wrap `dataclasses.field()` initiator for classes with additional structural metadata (like `BatchFlattenedSeq`).
#  That would allow defining these structural meta-parameters once in the tensors-data-class definition instead of
#    within each instance.
#  Of course it should also be supported by the relevant methods (like `collate()`). These params would have to be
#   propagated towards the nested collate() call and be used there.


@dataclasses.dataclass
class TensorsDataClass:
    _batch_size: Optional[int] = dataclasses.field(init=False, default=None)

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        supers = super(TensorsDataClass, cls).get_management_fields() \
            if hasattr(super(TensorsDataClass, cls), 'get_management_fields') else ()
        return supers + ('_batch_size', )

    @classmethod
    def get_data_fields(cls) -> Tuple[dataclasses.Field, ...]:
        return tuple(field for field in dataclasses.fields(cls) if field.name not in set(cls.get_management_fields()))

    @property
    def is_batched(self) -> int:
        return self._batch_size is not None

    @property
    def batch_size(self) -> int:
        if self._batch_size is None:
            raise ValueError(
                f'called `batch_size()` on a not batched `{self.__class__.__name__}(TensorsDataClass)`. '
                f'object id: {id(self)}')
        return self._batch_size

    @property
    def nr_examples(self):
        return self.batch_size  # for flattened batches it should be overridden

    def to(self, device):
        return self.deep_map(
            map_fn=lambda field_val: field_val.to(device) if hasattr(field_val, 'to') else field_val,
            mapper_override_group='device')

    def lazy_to(self, device, lazy_usage_history=None):
        map_fn = lambda field_val: field_val.to(device) if hasattr(field_val, 'to') else field_val
        return self.deep_lazy_map(
            map_fn=map_fn, mapper_override_group='device', lazy_map_usage_history=lazy_usage_history)

    def cpu(self):
        return self.deep_map(
            map_fn=lambda field_val: field_val.cpu() if hasattr(field_val, 'cpu') else field_val,
            mapper_override_group='device')

    def numpy(self):
        return self.cpu().deep_map(
            map_fn=lambda field_val: field_val.numpy() if hasattr(field_val, 'numpy') else field_val)

    def tolist(self):
        return self.cpu().deep_map(
            map_fn=lambda field_val: field_val.tolist() if hasattr(field_val, 'tolist') else field_val)

    def deep_map(
            self,
            map_fn: MapFn,
            mapper_override_group: Optional[str] = None,
            parents_path: Tuple['TensorsDataClass', ...] = (),
            fields_path: Tuple[str, ...] = ()) -> 'TensorsDataClass':
        mapped_field_values = {}
        new_parents_path = parents_path + (self,)
        for field in dataclasses.fields(self):
            field_value = object.__getattribute__(self, field.name)
            new_fields_path = fields_path + (field.name,)
            if isinstance(field_value, TensorsDataClass):
                assert not hasattr(self, '_lazy_map_fns_per_field') or field.name not in self._lazy_map_fns_per_field
                mapped_field_values[field.name] = field_value.deep_map(
                    map_fn=map_fn, mapper_override_group=mapper_override_group,
                    parents_path=new_parents_path, fields_path=new_fields_path)
            elif hasattr(self, '_lazy_map_fns_per_field') and field.name in self._lazy_map_fns_per_field:
                new_mapping_fns = list(self._lazy_map_fns_per_field[field.name])
                if mapper_override_group is not None:
                    while len(new_mapping_fns) > 0 and new_mapping_fns[-1][1] is not None and \
                            new_mapping_fns[-1][1] == mapper_override_group:
                        new_mapping_fns.pop()
                new_mapping_fns.append((map_fn, mapper_override_group))
                composed_map_fn = compose_fns(*(fn for fn, _ in new_mapping_fns))
                mapped_field_values[field.name] = composed_map_fn(field_value)
            else:
                mapped_field_values[field.name] = map_fn(field_value)
        new_obj = self.__class__(
            **{field.name: mapped_field_values[field.name] for field in dataclasses.fields(self) if field.init})
        for field in dataclasses.fields(self):
            if not field.init:
                setattr(new_obj, field.name, mapped_field_values[field.name])
        return new_obj

    def deep_lazy_map(
            self,
            map_fn: MapFn,
            mapper_override_group: Optional[str] = None,
            lazy_map_usage_history: Dict[Tuple[str, ...], Set[str]] = None,
            parents_path: Tuple['TensorsDataClass', ...] = (),
            fields_path: Tuple[str, ...] = ()) -> 'TensorsDataClass':
        assert '_lazy_map_fns_per_field' not in (field.name for field in dataclasses.fields(self))
        if lazy_map_usage_history is None:
            lazy_map_usage_history = {}
        mapped_field_values = {}
        lazy_map_fns_per_field: Dict[str, List[Tuple[MapFn, Optional[str]]]] = {}
        new_parents_path = parents_path + (self,)
        if fields_path not in lazy_map_usage_history:
            lazy_map_usage_history[fields_path] = set()
        lazy_map_usage_history_for_obj = lazy_map_usage_history[fields_path]
        for field in dataclasses.fields(self):
            field_value = object.__getattribute__(self, field.name)
            new_fields_path = fields_path + (field.name,)
            if isinstance(field_value, TensorsDataClass):
                assert field.name not in lazy_map_usage_history_for_obj
                assert not hasattr(self, '_lazy_map_fns_per_field') or field.name not in self._lazy_map_fns_per_field
                mapped_field_values[field.name] = field_value.deep_lazy_map(
                    map_fn=map_fn, mapper_override_group=mapper_override_group,
                    lazy_map_usage_history=lazy_map_usage_history,
                    parents_path=new_parents_path, fields_path=new_fields_path)
            elif field.name in lazy_map_usage_history_for_obj:
                assert not hasattr(self, '_lazy_map_fns_per_field') or field.name not in self._lazy_map_fns_per_field
                mapped_field_values[field.name] = map_fn(field_value)
            else:
                mapped_field_values[field.name] = field_value
                if hasattr(self, '_lazy_map_fns_per_field') and field.name in self._lazy_map_fns_per_field:
                    new_mapping_fns = list(self._lazy_map_fns_per_field[field.name])
                    if mapper_override_group is not None:
                        while len(new_mapping_fns) > 0 and new_mapping_fns[-1][1] is not None and \
                                new_mapping_fns[-1][1] == mapper_override_group:
                            new_mapping_fns.pop()
                    new_mapping_fns.append((map_fn, mapper_override_group))
                    lazy_map_fns_per_field[field.name] = new_mapping_fns
                else:
                    lazy_map_fns_per_field[field.name] = [(map_fn, mapper_override_group)]
        new_obj = self.__class__(
            **{field.name: mapped_field_values[field.name] for field in dataclasses.fields(self) if field.init})
        for field in dataclasses.fields(self):
            if not field.init:
                setattr(new_obj, field.name, mapped_field_values[field.name])
        new_obj._lazy_map_fns_per_field = lazy_map_fns_per_field
        new_obj._lazy_map_usage_history = lazy_map_usage_history_for_obj
        return new_obj

    def __getattribute__(self, field_name):
        # `__getattribute__` is overridden to support `lazy_map()`.
        if field_name == '_lazy_map_fns_per_field':
            return object.__getattribute__(self, field_name)
        if hasattr(self, '_lazy_map_fns_per_field') and field_name in self._lazy_map_fns_per_field:
            old_val = object.__getattribute__(self, field_name)
            composed_map_fn = compose_fns(*(fn for fn, _ in self._lazy_map_fns_per_field[field_name]))
            setattr(self, field_name, composed_map_fn(old_val))
            del self._lazy_map_fns_per_field[field_name]
            if len(self._lazy_map_fns_per_field) == 0:
                del self._lazy_map_fns_per_field
            self._lazy_map_usage_history.add(field_name)
        return object.__getattribute__(self, field_name)

    @classmethod
    def collate_values(cls, values_as_tuple: CollatableValuesTuple, collate_data: CollateData):
        assert all(type(value) == type(values_as_tuple[0]) for value in values_as_tuple)
        if values_as_tuple[0] is None:
            return None
        if isinstance(values_as_tuple[0], TensorsDataClass):
            assert hasattr(values_as_tuple[0].__class__, 'collate')
            return values_as_tuple[0].__class__.collate(
                values_as_tuple, collate_data=collate_data, is_most_outer_call=False)
        if isinstance(values_as_tuple[0], dict):
            all_keys = {key for dct in values_as_tuple for key in dct.keys()}
            return {key: cls.collate_values(tuple(dct[key] for dct in values_as_tuple if key in dct),
                                            collate_data=collate_data)
                    for key in all_keys}
        if isinstance(values_as_tuple[0], (list, tuple)):
            collection_type = type(values_as_tuple[0])
            assert all(len(lst) == len(values_as_tuple[0]) for lst in values_as_tuple)
            return collection_type(
                cls.collate_values(vals, collate_data=collate_data)
                for vals in zip(*values_as_tuple))
        if isinstance(values_as_tuple[0], torch.Tensor):
            return cls.collate_tensors(values_as_tuple)
        return values_as_tuple

    @classmethod
    def collate_tensors(cls, tensors: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if any(tensor.size() != tensors[0].size() for tensor in tensors):
            raise ValueError(
                f'Not all input tensors have the same tensor size. '
                f'sizes: {[tensor.size() for tensor in tensors]}')
        return torch.cat(tuple(tensor.unsqueeze(0) for tensor in tensors), dim=0)

    @classmethod
    def collate(cls, inputs: List['TensorsDataClass'],
                collate_data: Optional[CollateData] = None,
                is_most_outer_call: bool = True):
        assert all(isinstance(inp, cls) for inp in inputs)
        assert all(not inp.is_batched for inp in inputs)
        assert len(inputs) > 0
        if collate_data is None:
            collate_data = CollateData()
        batched_obj = cls._collate_first_pass(
            inputs, collate_data=collate_data)
        if batched_obj._batch_size is None:
            batched_obj._batch_size = len(inputs)
        if is_most_outer_call:
            batched_obj.post_collate_indices_fix((), (), collate_data)
            batched_obj.post_collate_remove_unnecessary_collate_info()
        return batched_obj

    @classmethod
    def _collate_first_pass(cls, inputs: List['TensorsDataClass'], collate_data: CollateData):
        batched_obj = cls(
            **{field.name: cls.collate_values(
                    tuple(getattr(inp, field.name) for inp in inputs),
                    collate_data=collate_data)
                for field in cls.get_data_fields() if field.init})
        batched_obj._batch_size = len(inputs)
        return batched_obj

    def post_collate_indices_fix(self, parents: Tuple['TensorsDataClass', ...], fields_path: Tuple[str, ...],
                                 collate_data: CollateData):
        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, TensorsDataClass):
                field_value.post_collate_indices_fix(
                    parents=parents + (self,), fields_path=fields_path + (field.name,),
                    collate_data=collate_data)

    def post_collate_remove_unnecessary_collate_info(self):
        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, TensorsDataClass):
                field_value.post_collate_remove_unnecessary_collate_info()

    def traverse(self,
                 parents_path: Optional[Tuple['TensorsDataClass', ...]] = None,
                 fields_path: Optional[Tuple[str, ...]] = None,
                 traversal_order: str = 'previsit'):
        assert traversal_order in ('previsit', 'postvisit')
        assert not ((parents_path is None) ^ (fields_path is None))
        if traversal_order == 'previsit':
            yield self if parents_path is None else (self, parents_path, fields_path)
        for field in self.get_data_fields():
            field_value = getattr(self, field.name)
            if not isinstance(field_value, TensorsDataClass):
                continue
            new_parents_path = None if parents_path is None else parents_path + (self,)
            new_fields_path = None if fields_path is None else fields_path + (field.name,)
            for child in field_value.traverse(
                    parents_path=new_parents_path, fields_path=new_fields_path, traversal_order=traversal_order):
                yield child
        if traversal_order == 'postvisit':
            yield self if parents_path is None else (self, parents_path, fields_path)



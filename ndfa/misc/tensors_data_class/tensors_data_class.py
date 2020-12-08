import torch
import dataclasses
from typing import List, Optional, Tuple, Dict, Set, Iterable, Union

try:
    from torch_geometric.data import Data as TorchGeometricData, Batch as TorchGeometricBatch
except ImportError:
    TorchGeometricData, TorchGeometricBatch = None, None

try:
    import dgl
except ImportError:
    dgl = None

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
    def get_indices_fields(cls) -> Tuple[dataclasses.Field, ...]:
        supers = super(TensorsDataClass, cls).get_indices_fields() \
            if hasattr(super(TensorsDataClass, cls), 'get_indices_fields') else ()
        return supers

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

    # TODO: generalize to support `TensorDataDict`!
    def deep_map(
            self,
            map_fn: MapFn,
            mapper_override_group: Optional[str] = None,
            parents_path: Tuple['TensorsDataClass', ...] = (),
            fields_path: Tuple[str, ...] = ()) -> 'TensorsDataClass':
        mapped_field_values = {}
        new_parents_path = parents_path + (self,)
        for field_name in self.get_field_names_by_group(group='all'):
            field_value = self.access_field_wo_applying_lazy_maps(field_name)
            new_fields_path = fields_path + (field_name,)
            if isinstance(field_value, TensorsDataClass):
                assert not hasattr(self, '_lazy_map_fns_per_field') or field_name not in self._lazy_map_fns_per_field
                mapped_field_values[field_name] = field_value.deep_map(
                    map_fn=map_fn, mapper_override_group=mapper_override_group,
                    parents_path=new_parents_path, fields_path=new_fields_path)
            elif hasattr(self, '_lazy_map_fns_per_field') and field_name in self._lazy_map_fns_per_field:
                new_mapping_fns = list(self._lazy_map_fns_per_field[field_name])
                if mapper_override_group is not None:
                    while len(new_mapping_fns) > 0 and new_mapping_fns[-1][1] is not None and \
                            new_mapping_fns[-1][1] == mapper_override_group:
                        new_mapping_fns.pop()
                new_mapping_fns.append((map_fn, mapper_override_group))
                composed_map_fn = compose_fns(*(fn for fn, _ in new_mapping_fns))
                mapped_field_values[field_name] = composed_map_fn(field_value)
            else:
                mapped_field_values[field_name] = map_fn(field_value)
        new_obj = self.factory(mapped_field_values)
        return new_obj

    # TODO: generalize to support `TensorDataDict`!
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
        for field_name in self.get_field_names_by_group(group='all'):
            field_value = self.access_field_wo_applying_lazy_maps(field_name)
            new_fields_path = fields_path + (field_name,)
            if isinstance(field_value, TensorsDataClass):
                assert field_name not in lazy_map_usage_history_for_obj
                assert not hasattr(self, '_lazy_map_fns_per_field') or field_name not in self._lazy_map_fns_per_field
                mapped_field_values[field_name] = field_value.deep_lazy_map(
                    map_fn=map_fn, mapper_override_group=mapper_override_group,
                    lazy_map_usage_history=lazy_map_usage_history,
                    parents_path=new_parents_path, fields_path=new_fields_path)
            # elif isinstance(field_value, dict):  # TODO: REMOVE this case! we do not longer support native dict!
            #     assert not hasattr(self, '_lazy_map_fns_per_field') or field_name not in self._lazy_map_fns_per_field
            #     mapped_field_values[field_name] = {
            #         key: val.deep_lazy_map(
            #             map_fn=map_fn, mapper_override_group=mapper_override_group,
            #             lazy_map_usage_history=lazy_map_usage_history,
            #             parents_path=new_parents_path, fields_path=new_fields_path + (key,))
            #             if isinstance(val, TensorsDataClass) else map_fn(val)  # TODO: fix it here..
            #         for key, val in field_value.items()}
            elif field_name in lazy_map_usage_history_for_obj:
                assert not hasattr(self, '_lazy_map_fns_per_field') or field_name not in self._lazy_map_fns_per_field
                mapped_field_values[field_name] = map_fn(field_value)
            else:
                mapped_field_values[field_name] = field_value
                if hasattr(self, '_lazy_map_fns_per_field') and field_name in self._lazy_map_fns_per_field:
                    new_mapping_fns = list(self._lazy_map_fns_per_field[field_name])
                    if mapper_override_group is not None:
                        while len(new_mapping_fns) > 0 and new_mapping_fns[-1][1] is not None and \
                                new_mapping_fns[-1][1] == mapper_override_group:
                            new_mapping_fns.pop()
                    new_mapping_fns.append((map_fn, mapper_override_group))
                    lazy_map_fns_per_field[field_name] = new_mapping_fns
                else:
                    lazy_map_fns_per_field[field_name] = [(map_fn, mapper_override_group)]

        new_obj = self.factory(mapped_field_values)
        new_obj._lazy_map_fns_per_field = lazy_map_fns_per_field
        new_obj._lazy_map_usage_history = lazy_map_usage_history_for_obj
        return new_obj

    @classmethod
    def factory(cls, kwargs: Dict) -> 'TensorsDataClass':
        dataclass_fields_not_to_init = {
            field.name: field for field in dataclasses.fields(cls) if not field.init}
        new_obj = cls(
            **{key: val for key, val in kwargs.items()
               if key not in dataclass_fields_not_to_init})
        for field_name in set(dataclass_fields_not_to_init.keys()) & set(kwargs.keys()):
            setattr(new_obj, field_name, kwargs[field_name])
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

    def access_field(self, name: str):
        return getattr(self, name)

    def access_field_wo_applying_lazy_maps(self, name: str):
        return object.__getattribute__(self, name)

    def get_all_fields(self) -> Tuple[str, ...]:
        return tuple(field.name for field in dataclasses.fields(self))

    def get_field_names_by_group(self, group: str = 'all') -> Tuple[str, ...]:
        if group == 'all':
            return self.get_all_fields()
        elif group == 'data':
            return tuple(field.name for field in self.get_data_fields())
        elif group == 'indices':
            return tuple(field.name for field in self.get_indices_fields())
        elif group == 'management':
            return tuple(self.get_management_fields())
        else:
            raise ValueError(f'Unsupported fields group `{group}`.')

    @classmethod
    def collate_values(cls, values_as_tuple: CollatableValuesTuple, collate_data: CollateData):
        assert all(type(value) == type(values_as_tuple[0]) for value in values_as_tuple)
        if values_as_tuple[0] is None:
            return None
        if isinstance(values_as_tuple[0], TensorsDataClass):
            assert hasattr(values_as_tuple[0].__class__, 'collate')
            return values_as_tuple[0].__class__.collate(
                values_as_tuple, collate_data=collate_data, is_most_outer_call=False)
        if TorchGeometricData is not None:
            if isinstance(values_as_tuple[0], TorchGeometricData):
                return TorchGeometricBatch.from_data_list(values_as_tuple, [])
            # TODO: just temporary. it should not be here.
            #  we should have a dedicated type `FlattenedTorchGeometricData`.
            if isinstance(values_as_tuple[0], (list, tuple)) and \
                    any(len(elem) > 0 and isinstance(elem[0], TorchGeometricData) for elem in values_as_tuple):
                flattened = tuple(values_as_tuple for elem in values_as_tuple for datum in elem)
                return TorchGeometricBatch.from_data_list(flattened, [])
        if dgl is not None:
            if isinstance(values_as_tuple[0], dgl.DGLGraph):
                return dgl.batch(list(values_as_tuple))
        # TODO: consider canceling native dict handling! (raise ValueError to use `TensorDataDict` instead)
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
                is_most_outer_call: bool = True) -> 'TensorsDataClass':
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
    def _collate_first_pass(cls, inputs: List['TensorsDataClass'], collate_data: CollateData) -> 'TensorsDataClass':
        batched_obj = cls(
            **{field.name: cls.collate_values(
                    tuple(getattr(inp, field.name) for inp in inputs),
                    collate_data=collate_data)
                for field in cls.get_data_fields() if field.init})
        batched_obj._batch_size = len(inputs)
        return batched_obj

    def post_collate_indices_fix(
            self, parents: Tuple['TensorsDataClass', ...],
            fields_path: Tuple[str, ...], collate_data: CollateData):
        for field_name in self.get_all_fields():
            field_value = self.access_field(field_name)
            if isinstance(field_value, TensorsDataClass):
                field_value.post_collate_indices_fix(
                    parents=parents + (self,), fields_path=fields_path + (field_name,),
                    collate_data=collate_data)

    def post_collate_remove_unnecessary_collate_info(self):
        for field_name in self.get_all_fields():
            field_value = self.access_field(field_name)
            if isinstance(field_value, TensorsDataClass):
                field_value.post_collate_remove_unnecessary_collate_info()

    def traverse(
            self, parents_path: Optional[Tuple['TensorsDataClass', ...]] = None,
            fields_path: Optional[Tuple[str, ...]] = None,
            traversal_order: str = 'previsit', fields_group: str = 'all') \
            -> Iterable[Union[
                'TensorsDataClass', Tuple['TensorsDataClass', Tuple['TensorsDataClass', ...], Tuple[str, ...]]]]:
        assert traversal_order in ('previsit', 'postvisit')
        assert not ((parents_path is None) ^ (fields_path is None))
        if traversal_order == 'previsit':
            yield self if parents_path is None else (self, parents_path, fields_path)
        for field_name in self.get_field_names_by_group(group=fields_group):
            field_value = self.access_field(field_name)
            if not isinstance(field_value, TensorsDataClass):
                continue
            new_parents_path = None if parents_path is None else parents_path + (self,)
            new_fields_path = None if fields_path is None else fields_path + (field_name,)
            for child in field_value.traverse(
                    parents_path=new_parents_path, fields_path=new_fields_path,
                    traversal_order=traversal_order, fields_group=fields_group):
                yield child
        if traversal_order == 'postvisit':
            yield self if parents_path is None else (self, parents_path, fields_path)

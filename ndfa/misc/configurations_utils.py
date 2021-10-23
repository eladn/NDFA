import abc
import enum
import typing
import base64
import hashlib
import argparse
import omegaconf
import dataclasses
from omegaconf import OmegaConf
from typing import TypeVar, Union, Container, Optional, Type, Tuple, List, Dict, Any, Collection, Callable


__all__ = ['create_argparser_from_dataclass_conf_structure', 'reinstantiate_omegaconf_container',
           'create_conf_dotlist_from_parsed_args', 'DispatchField', 'HasDispatchableField',
           'conf_field', 'get_conf_field_info', 'ConfFieldInfo', 'DeterministicallyHashable']


@dataclasses.dataclass
class OriginalTypeInfo:
    original_type: Type
    unwrapped_type: Optional[Type] = None
    is_optional: bool = False


def _get_original_type(_type: Type) -> OriginalTypeInfo:
    origin_type = typing.get_origin(_type)
    if origin_type is None:
        return OriginalTypeInfo(original_type=_type)
    if origin_type == typing.Union or _type == typing.Union:
        union_types = typing.get_args(_type)
        assert len(union_types) == 2 and union_types[1] == type(None)
        original_type_info = _get_original_type(union_types[0])
        original_type_info.is_optional = True
        return original_type_info
    assert typing.get_origin(origin_type) is None
    return OriginalTypeInfo(original_type=origin_type, unwrapped_type=_type)


def create_argparser_from_dataclass_conf_structure(
        _type: Type, argparser: Optional[argparse.ArgumentParser] = None, prefix: Tuple[str, ...] = ()) \
        -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser() if argparser is None else argparser
    original_field_type_info = _get_original_type(_type)
    assert dataclasses.is_dataclass(original_field_type_info.original_type)
    for field in dataclasses.fields(original_field_type_info.original_type):
        if field.type is None:
            continue
        arg_dest = '___'.join(prefix + (field.name,))
        arg_name = '--' + '.'.join(prefix + (field.name,))
        arg_negate_name = '--' + '.'.join(prefix + ('no_' + field.name,))
        original_field_type_info = _get_original_type(field.type)
        required = not original_field_type_info.is_optional and \
                   (field.default is dataclasses.MISSING or field.default is omegaconf.MISSING) and \
                   (field.default_factory is dataclasses.MISSING or field.default_factory is omegaconf.MISSING)

        conf_field_info = get_conf_field_info(field)
        if conf_field_info is None:
            conf_field_info = ConfFieldInfo()
        assert conf_field_info.description is None or isinstance(conf_field_info.description, str)

        if dataclasses.is_dataclass(original_field_type_info.original_type):
            create_argparser_from_dataclass_conf_structure(
                _type=field.type, argparser=argparser, prefix=prefix + (field.name,))
        elif issubclass(original_field_type_info.original_type, bool):
            group = argparser.add_mutually_exclusive_group(required=required)
            group.add_argument(
                arg_name, action='store_true', dest=arg_dest, default=None,
                help=conf_field_info.description)
            group.add_argument(
                arg_negate_name, action='store_false', dest=arg_dest, default=None,
                help=conf_field_info.description)
        elif issubclass(original_field_type_info.original_type, (int, float, str)):
            argparser.add_argument(
                arg_name, type=original_field_type_info.original_type,
                choices=conf_field_info.choices, required=required,
                dest=arg_dest, help=conf_field_info.description)
        elif original_field_type_info.original_type == typing.Literal:
            choices = typing.get_args(original_field_type_info.unwrapped_type)
            assert conf_field_info.choices is None or set(conf_field_info.choices) == set(choices)
            argparser.add_argument(
                arg_name, choices=choices,
                required=required, dest=arg_dest, help=conf_field_info.description)
        elif issubclass(original_field_type_info.original_type, enum.Enum):
            choices = tuple(original_field_type_info.original_type.__members__.keys())
            assert conf_field_info.choices is None or set(conf_field_info.choices) == set(choices)
            argparser.add_argument(
                arg_name, choices=choices,
                required=required, dest=arg_dest, help=conf_field_info.description)
        elif issubclass(original_field_type_info.original_type, (list, tuple, set, frozenset)):
            item_type = None
            if original_field_type_info.unwrapped_type is not None:
                container_typing_args = typing.get_args(original_field_type_info.unwrapped_type)
                assert (original_field_type_info.original_type != tuple and len(container_typing_args) <= 1) or \
                       (original_field_type_info.original_type == tuple and len(container_typing_args) == 0) or \
                       (original_field_type_info.original_type == tuple and len(container_typing_args) == 2 and container_typing_args[1] is ...)
                if len(container_typing_args) >= 1:
                    item_type = container_typing_args[0]

            # TODO: we currently don't cast to the wanted type because it seems that OmegaConf dotlist
            #  accept only lists (at least how we currently use it).
            #  However, it doesn't really matter, because we always perform `reinstantiate_omegaconf_container()`
            #  that re-creates the correct collection type.
            # class castContainerTypeAction(argparse.Action):
            #     def __call__(self, parser, args, values, option_string=None):
            #         values = original_field_type_info.original_type(values)
            #         setattr(args, self.dest, values)

            assert conf_field_info.choices is None
            argparser.add_argument(
                arg_name, choices=conf_field_info.elements_choices,
                type=item_type, nargs='+' if required else '*',  # action=castContainerTypeAction,
                required=required, dest=arg_dest, help=conf_field_info.description)
        else:
            assert False
        # TODO: support `ConfFieldInfo` additional info (arg_prefix, arg_names, ...)

    return argparser


SomeTypeT = TypeVar('SomeTypeT')


def reinstantiate_omegaconf_container(
        cnf: Union[Container, int, str, float], _type: Optional[Type[SomeTypeT]] = None) -> SomeTypeT:
    if cnf is None:
        return None
    if _type is None:
        _type = OmegaConf.get_type(cnf)
    original_type_info = _get_original_type(_type)
    if dataclasses.is_dataclass(original_type_info.original_type):
        field_values = {
            field.name: reinstantiate_omegaconf_container(
                OmegaConf.select(cnf, field.name, throw_on_missing=False), field.type)
            for field in dataclasses.fields(original_type_info.original_type)
            if not OmegaConf.is_missing(cnf, field.name)}
        return original_type_info.original_type(**field_values)
    elif issubclass(original_type_info.original_type, (list, tuple, set, frozenset)):
        item_type = None
        if original_type_info.unwrapped_type is not None:
            container_typing_args = typing.get_args(original_type_info.unwrapped_type)
            assert (original_type_info.original_type != tuple and len(container_typing_args) <= 1) or \
                   (original_type_info.original_type == tuple and len(container_typing_args) == 0) or \
                   (original_type_info.original_type == tuple and len(container_typing_args) == 2 and
                    container_typing_args[1] is ...)
            if len(container_typing_args) >= 1:
                item_type = container_typing_args[0]
        return original_type_info.original_type((reinstantiate_omegaconf_container(item, item_type) for item in cnf))
    elif original_type_info.original_type == dict:
        key_type, value_type = None, None
        if original_type_info.unwrapped_type is not None:
            container_typing_args = typing.get_args(original_type_info.unwrapped_type)
            assert len(container_typing_args) in {0, 2}
            if len(container_typing_args) == 2:
                key_type, value_type = container_typing_args
        return original_type_info.original_type({
            reinstantiate_omegaconf_container(item, key_type): reinstantiate_omegaconf_container(item, value_type)
            for item in cnf})
    elif issubclass(original_type_info.original_type, enum.Enum):
        chosen_enum_member = next(
            (member for member in original_type_info.original_type.__members__.values()
             if member == cnf or member.name == cnf), None)
        return chosen_enum_member
    assert issubclass(original_type_info.original_type, (int, str, float, bool))
    return original_type_info.original_type(cnf)


def create_conf_dotlist_from_parsed_args(args: argparse.Namespace) -> List[str]:
    dotlist = []
    for key, value in vars(args).items():
        if value is None:
            continue
        dotlist.append(f'{key.replace("___", ".")}={value}')
    return dotlist


@dataclasses.dataclass
class DispatchField:
    dispatch_field_name: str
    value_to_field_name_map: Dict[Any, Union[str, Collection[str]]]


@dataclasses.dataclass
class HasDispatchableField(abc.ABC):
    # _dispatch_fields: Dict[str, DispatchField] = field(default_factory=dict, init=False, compare=False, repr=False)

    @classmethod
    def register_dispatch_field(cls, dispatch_field: DispatchField):
        if not hasattr(cls, '_dispatch_fields'):
            cls._dispatch_fields: Dict[str, DispatchField] = {}
        if dispatch_field.dispatch_field_name in cls._dispatch_fields:
            return  # dispatch field already set
        dispatch_field = DispatchField(
            dispatch_field_name=dispatch_field.dispatch_field_name,
            value_to_field_name_map={
                val: {flds} if isinstance(flds, str) else set(flds)
                for val, flds in dispatch_field.value_to_field_name_map.items()})
        self_field_names = set(fld.name for fld in dataclasses.fields(cls))
        assert dispatch_field.dispatch_field_name in self_field_names
        assert all(fld_name in self_field_names
                   for fld_names in dispatch_field.value_to_field_name_map.values()
                   for fld_name in fld_names)
        cls._dispatch_fields[dispatch_field.dispatch_field_name] = dispatch_field

    @classmethod
    @abc.abstractmethod
    def set_dispatch_fields(cls):
        ...

    @classmethod
    def set_dispatch_fields_once(cls):
        if not hasattr(cls, '_dispatch_fields_set'):
            return
        cls.set_dispatch_fields()

    def __post_init__(self):
        super_cls = super(HasDispatchableField, self)
        if hasattr(super_cls, '__post_init__'):
            super_cls.__post_init__()
        self.set_dispatch_fields()

    def fix_dispatch_fields(self):
        # TODO: warn if we remove field that is set explicitly (not by default c'tor)
        if isinstance(self, HasDispatchableField):
            for dispatch_field in self._dispatch_fields.values():
                val = getattr(self, dispatch_field.dispatch_field_name)
                fields_to_keep = set()
                if val in dispatch_field.value_to_field_name_map:
                    to_keep = dispatch_field.value_to_field_name_map[val]
                    fields_to_keep = {to_keep} if isinstance(to_keep, str) else set(to_keep)
                all_dispatched_fields = set(fld for flds_group in dispatch_field.value_to_field_name_map.values() for fld in flds_group)
                fields_to_remove = all_dispatched_fields - fields_to_keep
                for field_name in fields_to_remove:
                    setattr(self, field_name, None)
        for fld in dataclasses.fields(self):
            val = getattr(self, fld.name, None)
            if val is not None and dataclasses.is_dataclass(val):
                HasDispatchableField.fix_dispatch_fields(val)


@dataclasses.dataclass
class ConfFieldInfo:
    description: Optional[str] = None
    default_as_other_field: Optional[str] = None
    default_factory_with_self_access: Optional[Callable] = None
    default_description: Optional[str] = None
    init_from_arg: bool = True
    arg_names: Optional[Collection[str]] = None
    arg_prefix: Optional[str] = None
    choices: Optional[Collection[Union[str, int, float]]] = None
    elements_choices: Optional[Collection[Union[str, int, float]]] = None


def conf_field(
        *, default=dataclasses.MISSING, default_factory=dataclasses.MISSING, init=True, repr=True,
        hash=None, compare=True, metadata=None, description: Optional[str] = None,
        arg_names: Optional[Collection[str]] = None, arg_prefix: Optional[str] = None,
        choices: Optional[Collection[Union[str, int, float]]] = None,
        elements_choices: Optional[Collection[Union[str, int, float]]] = None):
    metadata = {} if metadata is None else metadata
    assert '__ConfFieldInfo' not in metadata
    metadata['__ConfFieldInfo'] = ConfFieldInfo(
        description=description,
        arg_names=arg_names,
        arg_prefix=arg_prefix,
        choices=choices,
        elements_choices=elements_choices)
    return dataclasses.field(
        default=default, default_factory=default_factory, init=init,
        repr=repr, hash=hash, compare=compare, metadata=metadata)


def get_conf_field_info(field: dataclasses.Field) -> Optional[ConfFieldInfo]:
    return field.metadata.get('__ConfFieldInfo', None)


class DeterministicallyHashable:
    def get_sha1_base64(self) -> str:
        return base64.urlsafe_b64encode(self.get_sha1()).strip().decode('ascii').strip('=')

    def get_sha1(self) -> bytes:
        self_repr = OmegaConf.to_yaml(OmegaConf.structured(self))
        # self_repr = repr(self)
        return hashlib.sha1(self_repr.encode('utf8')).digest()

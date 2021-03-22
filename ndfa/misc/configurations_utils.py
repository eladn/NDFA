import enum
import typing
import argparse
import omegaconf
import dataclasses
from omegaconf import OmegaConf
from typing import TypeVar, Union, Container, Optional, Type, Tuple, List


__all__ = ['create_argparser_from_dataclass_conf_structure', 'reinstantiate_omegaconf_container',
           'create_conf_dotlist_from_parsed_args']


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
        if dataclasses.is_dataclass(original_field_type_info.original_type):
            create_argparser_from_dataclass_conf_structure(
                _type=field.type, argparser=argparser, prefix=prefix + (field.name,))
        elif issubclass(original_field_type_info.original_type, bool):
            group = argparser.add_mutually_exclusive_group(required=required)
            group.add_argument(arg_name, action='store_true', dest=arg_dest, default=None)
            group.add_argument(arg_negate_name, action='store_false', dest=arg_dest, default=None)
        elif issubclass(original_field_type_info.original_type, (int, float, str)):
            argparser.add_argument(
                arg_name, type=original_field_type_info.original_type,
                required=required, dest=arg_dest)
        elif original_field_type_info.original_type == typing.Literal:
            argparser.add_argument(
                arg_name, choices=typing.get_args(original_field_type_info.unwrapped_type),
                required=required, dest=arg_dest)
        elif issubclass(original_field_type_info.original_type, enum.Enum):
            argparser.add_argument(
                arg_name, choices=tuple(original_field_type_info.original_type.__members__.keys()),
                required=required, dest=arg_dest)
        elif issubclass(original_field_type_info.original_type, (list, tuple, set, frozenset)):
            item_type = None
            if original_field_type_info.unwrapped_type is not None:
                container_typing_args = typing.get_args(original_field_type_info.unwrapped_type)
                assert len(container_typing_args) <= 1
                if len(container_typing_args) == 1:
                    item_type = container_typing_args[0]
            # TODO: finish this case
        # TODO: support `confparam` meta-data (description & choices)

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
            assert len(container_typing_args) <= 1
            if len(container_typing_args) == 1:
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

# -*- coding: utf-8 -*-

from enum import Enum
from typing import Dict, Type, TypeVar

_EnumT = TypeVar("_EnumT", bound=Enum)


def make_name_map(enum_cls: Type[_EnumT], upper=True) -> Dict[str, _EnumT]:
    result = dict()
    for key, value in enum_cls.__members__.items():
        key = key.upper() if upper else key
        result[key] = value
    return result

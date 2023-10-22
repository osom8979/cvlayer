# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Union


@unique
class TextOrigin(Enum):
    BOTTOM_LEFT = True
    TOP_LEFT = False


def normalize_text_origin(text_origin: Union[TextOrigin, int, bool]) -> bool:
    if isinstance(text_origin, TextOrigin):
        return text_origin.value
    elif isinstance(text_origin, int):
        return text_origin != 0
    elif isinstance(text_origin, bool):
        return text_origin
    else:
        return bool(text_origin)

# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

TEXT_ORIGIN_BOTTOM_LEFT: Final[bool] = True
TEXT_ORIGIN_TOP_LEFT: Final[bool] = False


@unique
class TextOrigin(Enum):
    BOTTOM_LEFT = TEXT_ORIGIN_BOTTOM_LEFT
    TOP_LEFT = TEXT_ORIGIN_TOP_LEFT


TextOriginLike = Union[TextOrigin, str, int, bool]

TEXT_ORIGIN_MAP: Final[Dict[str, TextOrigin]] = {e.name.upper(): e for e in TextOrigin}
DEFAULT_TEXT_ORIGIN: Final[TextOriginLike] = TextOrigin.TOP_LEFT


def normalize_text_origin(text_origin: Optional[TextOriginLike]) -> bool:
    if text_origin is None:
        return TEXT_ORIGIN_TOP_LEFT

    if isinstance(text_origin, TextOrigin):
        return text_origin.value
    elif isinstance(text_origin, str):
        return TEXT_ORIGIN_MAP[text_origin.upper()].value
    elif isinstance(text_origin, int):
        return text_origin != 0
    elif isinstance(text_origin, bool):
        return text_origin
    else:
        raise TypeError(f"Unsupported text_origin type: {type(text_origin).__name__}")

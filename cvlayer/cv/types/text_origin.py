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

DEFAULT_TEXT_ORIGIN: Final[TextOriginLike] = TEXT_ORIGIN_TOP_LEFT
TEXT_ORIGIN_MAP: Final[Dict[str, bool]] = {
    # TextOrigin enum names
    "BOTTOM_LEFT": TEXT_ORIGIN_BOTTOM_LEFT,
    "TOP_LEFT": TEXT_ORIGIN_TOP_LEFT,
    # global constant names
    "TEXT_ORIGIN_BOTTOM_LEFT": TEXT_ORIGIN_BOTTOM_LEFT,
    "TEXT_ORIGIN_TOP_LEFT": TEXT_ORIGIN_TOP_LEFT,
}


def normalize_text_origin(text_origin: Optional[TextOriginLike]) -> bool:
    if text_origin is None:
        assert isinstance(DEFAULT_TEXT_ORIGIN, bool)
        return DEFAULT_TEXT_ORIGIN  # type: ignore[return-value]  # mypy bug ?

    if isinstance(text_origin, TextOrigin):
        return text_origin.value
    elif isinstance(text_origin, str):
        return TEXT_ORIGIN_MAP[text_origin.upper()]
    elif isinstance(text_origin, int):
        return text_origin != 0
    elif isinstance(text_origin, bool):
        return text_origin
    else:
        raise TypeError(f"Unsupported text_origin type: {type(text_origin).__name__}")

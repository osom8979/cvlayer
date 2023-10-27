# -*- coding: utf-8 -*-

from enum import Enum
from typing import Dict, Final, Optional, Union

import cv2

BORDER_CONSTANT: Final[int] = cv2.BORDER_CONSTANT
BORDER_REPLICATE: Final[int] = cv2.BORDER_REPLICATE
BORDER_REFLECT: Final[int] = cv2.BORDER_REFLECT
BORDER_WRAP: Final[int] = cv2.BORDER_WRAP
BORDER_REFLECT101: Final[int] = cv2.BORDER_REFLECT101
BORDER_TRANSPARENT: Final[int] = cv2.BORDER_TRANSPARENT
BORDER_DEFAULT: Final[int] = cv2.BORDER_DEFAULT
BORDER_ISOLATED: Final[int] = cv2.BORDER_ISOLATED


class BorderType(Enum):
    """Pixel extrapolation method"""

    CONSTANT = BORDER_CONSTANT
    REPLICATE = BORDER_REPLICATE
    REFLECT = BORDER_REFLECT
    WRAP = BORDER_WRAP
    REFLECT101 = BORDER_REFLECT101
    TRANSPARENT = BORDER_TRANSPARENT
    DEFAULT = BORDER_DEFAULT
    ISOLATED = BORDER_ISOLATED


BorderTypeLike = Union[BorderType, str, int]

DEFAULT_BORDER_TYPE: Final[BorderTypeLike] = BORDER_DEFAULT
BORDER_TYPE_MAP: Final[Dict[str, int]] = {
    # BorderType enum names
    "CONSTANT": BORDER_CONSTANT,
    "REPLICATE": BORDER_REPLICATE,
    "REFLECT": BORDER_REFLECT,
    "WRAP": BORDER_WRAP,
    "REFLECT101": BORDER_REFLECT101,
    "TRANSPARENT": BORDER_TRANSPARENT,
    "DEFAULT": BORDER_DEFAULT,
    "ISOLATED": BORDER_ISOLATED,
    # cv2 symbol full names
    "BORDER_CONSTANT": BORDER_CONSTANT,
    "BORDER_REPLICATE": BORDER_REPLICATE,
    "BORDER_REFLECT": BORDER_REFLECT,
    "BORDER_WRAP": BORDER_WRAP,
    "BORDER_REFLECT101": BORDER_REFLECT101,
    "BORDER_TRANSPARENT": BORDER_TRANSPARENT,
    "BORDER_DEFAULT": BORDER_DEFAULT,
    "BORDER_ISOLATED": BORDER_ISOLATED,
}

assert BORDER_DEFAULT == BORDER_REFLECT101
assert BorderType.DEFAULT.value == BorderType.REFLECT101.value


def normalize_border_type(border_type: Optional[BorderTypeLike]) -> int:
    if border_type is None:
        assert isinstance(DEFAULT_BORDER_TYPE, int)
        return DEFAULT_BORDER_TYPE

    if isinstance(border_type, BorderType):
        return border_type.value
    elif isinstance(border_type, str):
        return BORDER_TYPE_MAP[border_type.upper()]
    elif isinstance(border_type, int):
        return border_type
    else:
        raise TypeError(f"Unsupported border type: {type(border_type).__name__}")

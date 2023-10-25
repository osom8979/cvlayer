# -*- coding: utf-8 -*-

from enum import Enum
from typing import Dict, Final, Optional, Union

import cv2

from cvlayer.enum.utils import make_name_map

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

BORDER_TYPE_MAP: Final[Dict[str, BorderType]] = make_name_map(BorderType)
DEFAULT_BORDER_TYPE: Final[BorderTypeLike] = BorderType.DEFAULT

assert cv2.BORDER_DEFAULT == cv2.BORDER_REFLECT101
assert BorderType.DEFAULT.value == BorderType.REFLECT101.value


def normalize_border(border: Optional[BorderTypeLike]) -> int:
    if border is None:
        return cv2.BORDER_DEFAULT

    if isinstance(border, BorderType):
        return border.value
    elif isinstance(border, str):
        return BORDER_TYPE_MAP[border.upper()].value
    elif isinstance(border, int):
        return border
    else:
        raise TypeError(f"Unsupported border type: {type(border).__name__}")

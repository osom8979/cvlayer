# -*- coding: utf-8 -*-

from enum import Enum
from typing import Dict, Final, Optional, Union

import cv2


class BorderType(Enum):
    """Pixel extrapolation method"""

    CONSTANT = cv2.BORDER_CONSTANT
    REPLICATE = cv2.BORDER_REPLICATE
    REFLECT = cv2.BORDER_REFLECT
    WRAP = cv2.BORDER_WRAP
    REFLECT101 = cv2.BORDER_REFLECT101
    TRANSPARENT = cv2.BORDER_TRANSPARENT
    DEFAULT = cv2.BORDER_DEFAULT
    ISOLATED = cv2.BORDER_ISOLATED


BorderTypeLike = Union[BorderType, str, int]

BORDER_TYPE_MAP: Final[Dict[str, BorderType]] = {e.name.upper(): e for e in BorderType}
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

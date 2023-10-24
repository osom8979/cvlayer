# -*- coding: utf-8 -*-

from enum import Enum
from typing import Final, Optional, Union

import cv2


class BorderType(Enum):
    CONSTANT = cv2.BORDER_CONSTANT
    REPLICATE = cv2.BORDER_REPLICATE
    REFLECT = cv2.BORDER_REFLECT
    WRAP = cv2.BORDER_WRAP
    REFLECT101 = cv2.BORDER_REFLECT101
    TRANSPARENT = cv2.BORDER_TRANSPARENT
    DEFAULT = cv2.BORDER_DEFAULT
    ISOLATED = cv2.BORDER_ISOLATED


DEFAULT_BORDER_TYPE: Final[Union[BorderType, int]] = BorderType.DEFAULT

assert cv2.BORDER_DEFAULT == cv2.BORDER_REFLECT101
assert cv2.BORDER_DEFAULT == BorderType.REFLECT101.value


def normalize_border(border: Optional[Union[BorderType, int]] = None) -> int:
    if border is None:
        return cv2.BORDER_DEFAULT
    elif isinstance(border, BorderType):
        return border.value
    elif isinstance(border, int):
        return border
    else:
        raise TypeError(f"Unsupported border type: {type(border).__name__}")

# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

import cv2

FONT_HERSHEY_SIMPLEX: Final[int] = cv2.FONT_HERSHEY_SIMPLEX
FONT_HERSHEY_PLAIN: Final[int] = cv2.FONT_HERSHEY_PLAIN
FONT_HERSHEY_DUPLEX: Final[int] = cv2.FONT_HERSHEY_DUPLEX
FONT_HERSHEY_COMPLEX: Final[int] = cv2.FONT_HERSHEY_COMPLEX
FONT_HERSHEY_TRIPLEX: Final[int] = cv2.FONT_HERSHEY_TRIPLEX
FONT_HERSHEY_COMPLEX_SMALL: Final[int] = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONT_HERSHEY_SCRIPT_SIMPLEX: Final[int] = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
FONT_HERSHEY_SCRIPT_COMPLEX: Final[int] = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
FONT_ITALIC: Final[int] = cv2.FONT_ITALIC


@unique
class HersheyFont(Enum):
    SIMPLEX: Final[int] = cv2.FONT_HERSHEY_SIMPLEX
    PLAIN: Final[int] = cv2.FONT_HERSHEY_PLAIN
    DUPLEX: Final[int] = cv2.FONT_HERSHEY_DUPLEX
    COMPLEX: Final[int] = cv2.FONT_HERSHEY_COMPLEX
    TRIPLEX: Final[int] = cv2.FONT_HERSHEY_TRIPLEX
    COMPLEX_SMALL: Final[int] = cv2.FONT_HERSHEY_COMPLEX_SMALL
    SCRIPT_SIMPLEX: Final[int] = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    SCRIPT_COMPLEX: Final[int] = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    ITALIC: Final[int] = cv2.FONT_ITALIC


HersheyFontLike = Union[HersheyFont, str, int]

HERSHEY_FONT_MAP: Final[Dict[str, HersheyFont]] = {
    e.name.upper(): e for e in HersheyFont
}
DEFAULT_HERSHEY_FONT: Final[HersheyFontLike] = HersheyFont.SIMPLEX


def normalize_font_face(font: Optional[HersheyFontLike]) -> int:
    if font is None:
        return FONT_HERSHEY_SIMPLEX

    if isinstance(font, HersheyFont):
        return font.value
    elif isinstance(font, str):
        return HERSHEY_FONT_MAP[font.upper()].value
    elif isinstance(font, int):
        return font
    else:
        raise TypeError(f"Unsupported font type: {type(font).__name__}")

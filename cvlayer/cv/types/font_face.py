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
    SIMPLEX = FONT_HERSHEY_SIMPLEX
    PLAIN = FONT_HERSHEY_PLAIN
    DUPLEX = FONT_HERSHEY_DUPLEX
    COMPLEX = FONT_HERSHEY_COMPLEX
    TRIPLEX = FONT_HERSHEY_TRIPLEX
    COMPLEX_SMALL = FONT_HERSHEY_COMPLEX_SMALL
    SCRIPT_SIMPLEX = FONT_HERSHEY_SCRIPT_SIMPLEX
    SCRIPT_COMPLEX = FONT_HERSHEY_SCRIPT_COMPLEX
    ITALIC = FONT_ITALIC


HersheyFontLike = Union[HersheyFont, str, int]

DEFAULT_HERSHEY_FONT: Final[HersheyFontLike] = FONT_HERSHEY_SIMPLEX
HERSHEY_FONT_MAP: Final[Dict[str, int]] = {
    # DistanceType enum names
    "SIMPLEX": FONT_HERSHEY_SIMPLEX,
    "PLAIN": FONT_HERSHEY_PLAIN,
    "DUPLEX": FONT_HERSHEY_DUPLEX,
    "COMPLEX": FONT_HERSHEY_COMPLEX,
    "TRIPLEX": FONT_HERSHEY_TRIPLEX,
    "COMPLEX_SMALL": FONT_HERSHEY_COMPLEX_SMALL,
    "SCRIPT_SIMPLEX": FONT_HERSHEY_SCRIPT_SIMPLEX,
    "SCRIPT_COMPLEX": FONT_HERSHEY_SCRIPT_COMPLEX,
    "ITALIC": FONT_ITALIC,
    # cv2 symbol suffix names
    "HERSHEY_SIMPLEX": FONT_HERSHEY_SIMPLEX,
    "HERSHEY_PLAIN": FONT_HERSHEY_PLAIN,
    "HERSHEY_DUPLEX": FONT_HERSHEY_DUPLEX,
    "HERSHEY_COMPLEX": FONT_HERSHEY_COMPLEX,
    "HERSHEY_TRIPLEX": FONT_HERSHEY_TRIPLEX,
    "HERSHEY_COMPLEX_SMALL": FONT_HERSHEY_COMPLEX_SMALL,
    "HERSHEY_SCRIPT_SIMPLEX": FONT_HERSHEY_SCRIPT_SIMPLEX,
    "HERSHEY_SCRIPT_COMPLEX": FONT_HERSHEY_SCRIPT_COMPLEX,
    # cv2 symbol full names
    "FONT_HERSHEY_SIMPLEX": FONT_HERSHEY_SIMPLEX,
    "FONT_HERSHEY_PLAIN": FONT_HERSHEY_PLAIN,
    "FONT_HERSHEY_DUPLEX": FONT_HERSHEY_DUPLEX,
    "FONT_HERSHEY_COMPLEX": FONT_HERSHEY_COMPLEX,
    "FONT_HERSHEY_TRIPLEX": FONT_HERSHEY_TRIPLEX,
    "FONT_HERSHEY_COMPLEX_SMALL": FONT_HERSHEY_COMPLEX_SMALL,
    "FONT_HERSHEY_SCRIPT_SIMPLEX": FONT_HERSHEY_SCRIPT_SIMPLEX,
    "FONT_HERSHEY_SCRIPT_COMPLEX": FONT_HERSHEY_SCRIPT_COMPLEX,
    "FONT_ITALIC": FONT_ITALIC,
}


def normalize_font_face(font: Optional[HersheyFontLike]) -> int:
    if font is None:
        assert isinstance(FONT_HERSHEY_SIMPLEX, int)
        return FONT_HERSHEY_SIMPLEX

    if isinstance(font, HersheyFont):
        return font.value
    elif isinstance(font, str):
        return HERSHEY_FONT_MAP[font.upper()]
    elif isinstance(font, int):
        return font
    else:
        raise TypeError(f"Unsupported font type: {type(font).__name__}")

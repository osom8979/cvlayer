# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final, Union

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


def normalize_font_face(font: Union[HersheyFont, int]) -> int:
    return font.value if isinstance(font, HersheyFont) else font

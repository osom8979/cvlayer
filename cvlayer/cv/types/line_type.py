# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

import cv2

LINE_4: Final[int] = cv2.LINE_4  # Bresenham 4 Connect
LINE_8: Final[int] = cv2.LINE_8  # Bresenham 8 Connect
LINE_AA: Final[int] = cv2.LINE_AA  # Anti-Aliasing


@unique
class LineType(Enum):
    B4 = LINE_4
    B8 = LINE_8
    AA = LINE_AA


LineTypeLike = Union[LineType, str, int]

DEFAULT_LINE_TYPE: Final[LineTypeLike] = LINE_8
LINE_TYPE_MAP: Final[Dict[str, int]] = {
    # str to int names
    "4": LINE_4,
    "8": LINE_8,
    # LineType enum names
    "B4": LINE_4,
    "B8": LINE_8,
    "AA": LINE_AA,
    # cv2 symbol full names
    "LINE_4": LINE_4,
    "LINE_8": LINE_8,
    "LINE_AA": LINE_AA,
}


def normalize_line_type(line: Optional[LineTypeLike]) -> int:
    if line is None:
        assert isinstance(DEFAULT_LINE_TYPE, int)
        return DEFAULT_LINE_TYPE

    if isinstance(line, LineType):
        return line.value
    elif isinstance(line, str):
        return LINE_TYPE_MAP[line.upper()]
    elif isinstance(line, int):
        return line
    else:
        raise TypeError(f"Unsupported line type: {type(line).__name__}")

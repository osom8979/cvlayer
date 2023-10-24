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

LINE_TYPE_MAP: Final[Dict[str, LineType]] = {e.name.upper(): e for e in LineType}
DEFAULT_LINE_TYPE: Final[LineTypeLike] = LineType.B8


def normalize_line(line: Optional[LineTypeLike]) -> int:
    if line is None:
        return LINE_8

    if isinstance(line, LineType):
        return line.value
    elif isinstance(line, str):
        return LINE_TYPE_MAP[line.upper()].value
    elif isinstance(line, int):
        return line
    else:
        raise TypeError(f"Unsupported line type: {type(line).__name__}")

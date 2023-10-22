# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final, Union

import cv2

LINE_4: Final[int] = cv2.LINE_4  # Bresenham 4 Connect
LINE_8: Final[int] = cv2.LINE_8  # Bresenham 8 Connect
LINE_AA: Final[int] = cv2.LINE_AA  # Anti-Aliasing


@unique
class LineType(Enum):
    B4 = LINE_4
    B8 = LINE_8
    AA = LINE_AA


def normalize_line(line: Union[LineType, int]) -> int:
    return line.value if isinstance(line, LineType) else line

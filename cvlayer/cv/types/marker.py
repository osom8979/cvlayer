# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Union

import cv2


@unique
class MarkerType(Enum):
    CROSS = cv2.MARKER_CROSS
    TILTED_CROSS = cv2.MARKER_TILTED_CROSS
    STAR = cv2.MARKER_STAR
    DIAMOND = cv2.MARKER_DIAMOND
    SQUARE = cv2.MARKER_SQUARE
    TRIANGLE_UP = cv2.MARKER_TRIANGLE_UP
    TRIANGLE_DOWN = cv2.MARKER_TRIANGLE_DOWN


def normalize_marker(marker: Union[MarkerType, int]) -> int:
    return marker.value if isinstance(marker, MarkerType) else marker

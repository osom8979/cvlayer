# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

import cv2

MARKER_CROSS: Final[int] = cv2.MARKER_CROSS
MARKER_TILTED_CROSS: Final[int] = cv2.MARKER_TILTED_CROSS
MARKER_STAR: Final[int] = cv2.MARKER_STAR
MARKER_DIAMOND: Final[int] = cv2.MARKER_DIAMOND
MARKER_SQUARE: Final[int] = cv2.MARKER_SQUARE
MARKER_TRIANGLE_UP: Final[int] = cv2.MARKER_TRIANGLE_UP
MARKER_TRIANGLE_DOWN: Final[int] = cv2.MARKER_TRIANGLE_DOWN


@unique
class MarkerType(Enum):
    CROSS = MARKER_CROSS
    TILTED_CROSS = MARKER_TILTED_CROSS
    STAR = MARKER_STAR
    DIAMOND = MARKER_DIAMOND
    SQUARE = MARKER_SQUARE
    TRIANGLE_UP = MARKER_TRIANGLE_UP
    TRIANGLE_DOWN = MARKER_TRIANGLE_DOWN


MarkerTypeLike = Union[MarkerType, str, int]

DEFAULT_MARKER_TYPE: Final[MarkerTypeLike] = MARKER_CROSS
MARKER_TYPE_MAP: Final[Dict[str, int]] = {
    # LineType enum names
    "CROSS": MARKER_CROSS,
    "TILTED_CROSS": MARKER_TILTED_CROSS,
    "STAR": MARKER_STAR,
    "DIAMOND": MARKER_DIAMOND,
    "SQUARE": MARKER_SQUARE,
    "TRIANGLE_UP": MARKER_TRIANGLE_UP,
    "TRIANGLE_DOWN": MARKER_TRIANGLE_DOWN,
    # cv2 symbol full names
    "MARKER_CROSS": MARKER_CROSS,
    "MARKER_TILTED_CROSS": MARKER_TILTED_CROSS,
    "MARKER_STAR": MARKER_STAR,
    "MARKER_DIAMOND": MARKER_DIAMOND,
    "MARKER_SQUARE": MARKER_SQUARE,
    "MARKER_TRIANGLE_UP": MARKER_TRIANGLE_UP,
    "MARKER_TRIANGLE_DOWN": MARKER_TRIANGLE_DOWN,
}


def normalize_marker(marker: Optional[MarkerTypeLike]) -> int:
    if marker is None:
        assert isinstance(DEFAULT_MARKER_TYPE, int)
        return DEFAULT_MARKER_TYPE

    if isinstance(marker, MarkerType):
        return marker.value
    elif isinstance(marker, str):
        return MARKER_TYPE_MAP[marker.upper()]
    elif isinstance(marker, int):
        return marker
    else:
        raise TypeError(f"Unsupported marker type: {type(marker).__name__}")

# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

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


MarkerTypeLike = Union[MarkerType, str, int]

MARKER_TYPE_MAP: Final[Dict[str, MarkerType]] = {e.name.upper(): e for e in MarkerType}
DEFAULT_MARKER_TYPE: Final[MarkerTypeLike] = MarkerType.CROSS


def normalize_marker(marker: Optional[MarkerTypeLike]) -> int:
    if marker is None:
        return cv2.MARKER_CROSS

    if isinstance(marker, MarkerType):
        return marker.value
    elif isinstance(marker, str):
        return MARKER_TYPE_MAP[marker.upper()].value
    elif isinstance(marker, int):
        return marker
    else:
        raise TypeError(f"Unsupported marker type: {type(marker).__name__}")

# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

import cv2

DIST_USER = cv2.DIST_USER
DIST_L1 = cv2.DIST_L1
DIST_L2 = cv2.DIST_L2
DIST_C = cv2.DIST_C
DIST_L12 = cv2.DIST_L12
DIST_FAIR = cv2.DIST_FAIR
DIST_WELSCH = cv2.DIST_WELSCH
DIST_HUBER = cv2.DIST_HUBER


@unique
class DistanceType(Enum):
    """Distance types for Distance Transform and M-estimators"""

    USER = DIST_USER
    """User defined distance"""

    L1 = DIST_L1
    """distance = |x1-x2| + |y1-y2|"""

    L2 = DIST_L2
    """the simple euclidean distance"""

    C = DIST_C
    """distance = max(|x1-x2|,|y1-y2|)"""

    L12 = DIST_L12
    """L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))"""

    FAIR = DIST_FAIR
    """distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998"""

    WELSCH = DIST_WELSCH
    """distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846"""

    HUBER = DIST_HUBER
    """distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345"""


DistanceTypeLike = Union[DistanceType, str, int]

DEFAULT_DISTANCE_TYPE: Final[DistanceTypeLike] = DIST_L12
DISTANCE_TYPE_MAP: Final[Dict[str, int]] = {
    # DistanceType enum names
    "USER": DIST_USER,
    "L1": DIST_L1,
    "L2": DIST_L2,
    "C": DIST_C,
    "L12": DIST_L12,
    "FAIR": DIST_FAIR,
    "WELSCH": DIST_WELSCH,
    "HUBER": DIST_HUBER,
    # cv2 symbol full names
    "DIST_USER": DIST_USER,
    "DIST_L1": DIST_L1,
    "DIST_L2": DIST_L2,
    "DIST_C": DIST_C,
    "DIST_L12": DIST_L12,
    "DIST_FAIR": DIST_FAIR,
    "DIST_WELSCH": DIST_WELSCH,
    "DIST_HUBER": DIST_HUBER,
}


def normalize_distance_type(distance_type: Optional[DistanceTypeLike]) -> int:
    if distance_type is None:
        assert isinstance(DEFAULT_DISTANCE_TYPE, int)
        return DEFAULT_DISTANCE_TYPE

    if isinstance(distance_type, DistanceType):
        return distance_type.value
    elif isinstance(distance_type, str):
        return DISTANCE_TYPE_MAP[distance_type.upper()]
    elif isinstance(distance_type, int):
        return distance_type
    else:
        raise TypeError(f"Unsupported distance type: {type(distance_type).__name__}")

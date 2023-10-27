# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

import cv2

DIST_MASK_3: Final[int] = cv2.DIST_MASK_3
DIST_MASK_5: Final[int] = cv2.DIST_MASK_5
DIST_MASK_PRECISE: Final[int] = cv2.DIST_MASK_PRECISE


@unique
class DistanceTransformMask(Enum):
    """Mask size for distance transform"""

    M3 = DIST_MASK_3
    """mask=3"""

    M5 = DIST_MASK_5
    """mask=5"""

    PRECISE = DIST_MASK_PRECISE


DistanceTransformMaskLike = Union[DistanceTransformMask, str, int]

DEFAULT_DISTANCE_TRANSFORM_MASK: Final[DistanceTransformMaskLike] = DIST_MASK_3
DISTANCE_TRANSFORM_MASK_MAP: Final[Dict[str, int]] = {
    # str to int names
    "3": DIST_MASK_3,
    "5": DIST_MASK_5,
    # DistanceType enum names
    "M3": DIST_MASK_3,
    "M5": DIST_MASK_5,
    "PRECISE": DIST_MASK_PRECISE,
    # cv2 symbol suffix names
    "MASK_3": DIST_MASK_3,
    "MASK_5": DIST_MASK_5,
    "MASK_PRECISE": DIST_MASK_PRECISE,
    # cv2 symbol full names
    "DIST_MASK_3": DIST_MASK_3,
    "DIST_MASK_5": DIST_MASK_5,
    "DIST_MASK_PRECISE": DIST_MASK_PRECISE,
}


def normalize_distance_transform_mask(mask: Optional[DistanceTransformMaskLike]) -> int:
    if mask is None:
        assert isinstance(DEFAULT_DISTANCE_TRANSFORM_MASK, int)
        return DEFAULT_DISTANCE_TRANSFORM_MASK

    if isinstance(mask, DistanceTransformMask):
        return mask.value
    elif isinstance(mask, str):
        return DISTANCE_TRANSFORM_MASK_MAP[mask.upper()]
    elif isinstance(mask, int):
        return mask
    else:
        raise TypeError(f"Unsupported mask type: {type(mask).__name__}")

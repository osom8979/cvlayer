# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

import cv2

INTER_NEAREST: Final[int] = cv2.INTER_NEAREST
INTER_LINEAR: Final[int] = cv2.INTER_LINEAR
INTER_CUBIC: Final[int] = cv2.INTER_CUBIC
INTER_AREA: Final[int] = cv2.INTER_AREA
INTER_LANCZOS4: Final[int] = cv2.INTER_LANCZOS4
INTER_LINEAR_EXACT: Final[int] = cv2.INTER_LINEAR_EXACT
INTER_NEAREST_EXACT: Final[int] = cv2.INTER_NEAREST_EXACT
INTER_MAX: Final[int] = cv2.INTER_MAX


@unique
class Interpolation(Enum):
    NEAREST = INTER_NEAREST
    LINEAR = INTER_LINEAR
    CUBIC = INTER_CUBIC
    AREA = INTER_AREA
    LANCZOS4 = INTER_LANCZOS4
    LINEAR_EXACT = INTER_LINEAR_EXACT
    NEAREST_EXACT = INTER_NEAREST_EXACT
    MAX = INTER_MAX


InterpolationLike = Union[Interpolation, str, int]

DEFAULT_INTERPOLATION: Final[InterpolationLike] = Interpolation.NEAREST
INTERPOLATION_MAP: Final[Dict[str, int]] = {
    # Interpolation enum names
    "NEAREST": INTER_NEAREST,
    "LINEAR": INTER_LINEAR,
    "CUBIC": INTER_CUBIC,
    "AREA": INTER_AREA,
    "LANCZOS4": INTER_LANCZOS4,
    "LINEAR_EXACT": INTER_LINEAR_EXACT,
    "NEAREST_EXACT": INTER_NEAREST_EXACT,
    "MAX": INTER_MAX,
    # cv2 symbol full names
    "INTER_NEAREST": INTER_NEAREST,
    "INTER_LINEAR": INTER_LINEAR,
    "INTER_CUBIC": INTER_CUBIC,
    "INTER_AREA": INTER_AREA,
    "INTER_LANCZOS4": INTER_LANCZOS4,
    "INTER_LINEAR_EXACT": INTER_LINEAR_EXACT,
    "INTER_NEAREST_EXACT": INTER_NEAREST_EXACT,
    "INTER_MAX": INTER_MAX,
}


def normalize_interpolation(inter: Optional[InterpolationLike]) -> int:
    if inter is None:
        assert isinstance(DEFAULT_INTERPOLATION, int)
        return DEFAULT_INTERPOLATION

    if isinstance(inter, Interpolation):
        return inter.value
    elif isinstance(inter, str):
        return INTERPOLATION_MAP[inter.upper()]
    elif isinstance(inter, int):
        return inter
    else:
        raise TypeError(f"Unsupported inter type: {type(inter).__name__}")

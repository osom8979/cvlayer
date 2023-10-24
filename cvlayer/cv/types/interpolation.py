# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

import cv2


@unique
class Interpolation(Enum):
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4
    LINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    NEAREST_EXACT = cv2.INTER_NEAREST_EXACT
    MAX = cv2.INTER_MAX


InterpolationLike = Union[Interpolation, str, int]

INTERPOLATION_MAP: Final[Dict[str, Interpolation]] = {
    e.name.upper(): e for e in Interpolation
}
DEFAULT_INTERPOLATION: Final[InterpolationLike] = Interpolation.NEAREST


def normalize_interpolation(inter: Optional[InterpolationLike]) -> int:
    if inter is None:
        return cv2.INTER_NEAREST

    if isinstance(inter, Interpolation):
        return inter.value
    elif isinstance(inter, str):
        return INTERPOLATION_MAP[inter.upper()].value
    elif isinstance(inter, int):
        return inter
    else:
        raise TypeError(f"Unsupported inter type: {type(inter).__name__}")

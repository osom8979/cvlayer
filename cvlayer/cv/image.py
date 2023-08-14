# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final

import cv2
from numpy.typing import NDArray


@unique
class Interpolation(Enum):
    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_AREA = cv2.INTER_AREA
    INTER_LANCZOS4 = cv2.INTER_LANCZOS4
    INTER_LINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    INTER_NEAREST_EXACT = cv2.INTER_NEAREST_EXACT
    INTER_MAX = cv2.INTER_MAX


DEFAULT_INTERPOLATION: Final[Interpolation] = Interpolation.INTER_NEAREST


def scale_image(
    src: NDArray,
    scale: float,
    interpolation=DEFAULT_INTERPOLATION,
) -> NDArray:
    return cv2.resize(
        src,
        dsize=(0, 0),
        fx=scale,  # noqa
        fy=scale,  # noqa
        interpolation=interpolation.value,
    )

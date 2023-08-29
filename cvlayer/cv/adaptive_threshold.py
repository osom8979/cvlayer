# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final

import cv2
from numpy.typing import NDArray

from cvlayer.cv.color import PIXEL_8BIT_MAX
from cvlayer.cv.threshold import ThresholdMethod


@unique
class AdaptiveMethod(Enum):
    MEAN = cv2.ADAPTIVE_THRESH_MEAN_C
    GAUSSIAN = cv2.ADAPTIVE_THRESH_GAUSSIAN_C


DEFAULT_BLOCK_SIZE: Final[int] = 15
DEFAULT_C: Final[int] = 0


def adaptive_threshold(
    src: NDArray,
    max_value=PIXEL_8BIT_MAX,
    adaptive_method=AdaptiveMethod.MEAN,
    method=ThresholdMethod.BINARY,
    block_size=DEFAULT_BLOCK_SIZE,
    c=DEFAULT_C,
) -> NDArray:
    return cv2.adaptiveThreshold(
        src, max_value, adaptive_method.value, method.value, block_size, c
    )

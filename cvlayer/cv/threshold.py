# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import NamedTuple

import cv2
from numpy.typing import NDArray

from cvlayer.cv.color import PIXEL_8BIT_HALF, PIXEL_8BIT_MAX


@unique
class ThresholdMethod(Enum):
    BINARY = cv2.THRESH_BINARY
    BINARY_INV = cv2.THRESH_BINARY_INV
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO
    TOZERO_INV = cv2.THRESH_TOZERO_INV


class ThresholdResult(NamedTuple):
    computed_threshold_value: float
    threshold_image: NDArray


def threshold(
    src: NDArray,
    thresh=PIXEL_8BIT_HALF,
    max_value=PIXEL_8BIT_MAX,
    method=ThresholdMethod.BINARY,
) -> ThresholdResult:
    # The computed threshold value if Otsu's or Triangle methods used.
    computed_threshold_value, threshold_image = cv2.threshold(
        src, thresh, max_value, method.value
    )
    return ThresholdResult(computed_threshold_value, threshold_image)

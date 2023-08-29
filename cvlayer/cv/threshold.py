# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final, NamedTuple

import cv2
from numpy.typing import NDArray

from cvlayer.cv.color import PIXEL_8BIT_HALF, PIXEL_8BIT_MAX

DEFAULT_BLOCK_SIZE: Final[int] = 15
DEFAULT_C: Final[int] = 0


@unique
class AdaptiveMethod(Enum):
    MEAN = cv2.ADAPTIVE_THRESH_MEAN_C
    GAUSSIAN = cv2.ADAPTIVE_THRESH_GAUSSIAN_C


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


class CvlThreshold:
    @staticmethod
    def cvl_threshold(
        src: NDArray,
        thresh=PIXEL_8BIT_HALF,
        max_value=PIXEL_8BIT_MAX,
        method=ThresholdMethod.BINARY,
    ):
        return threshold(src, thresh, max_value, method)

    @staticmethod
    def cvl_adaptive_threshold(
        src: NDArray,
        max_value=PIXEL_8BIT_MAX,
        adaptive_method=AdaptiveMethod.MEAN,
        method=ThresholdMethod.BINARY,
        block_size=DEFAULT_BLOCK_SIZE,
        c=DEFAULT_C,
    ):
        return adaptive_threshold(
            src, max_value, adaptive_method, method, block_size, c
        )

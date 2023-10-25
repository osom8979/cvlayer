# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final, NamedTuple, Sequence

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
    """
    The computed threshold value if Otsu's or Triangle methods used.
    """

    threshold_image: NDArray


ADAPTIVE_THRESHOLD_METHODS: Final[Sequence[ThresholdMethod]] = (
    ThresholdMethod.BINARY,
    ThresholdMethod.BINARY_INV,
)


def threshold(
    src: NDArray,
    thresh=PIXEL_8BIT_HALF,
    max_value=PIXEL_8BIT_MAX,
    method=ThresholdMethod.BINARY,
) -> ThresholdResult:
    computed_threshold_value, threshold_image = cv2.threshold(
        src, thresh, max_value, method.value
    )
    return ThresholdResult(computed_threshold_value, threshold_image)


def threshold_otsu(
    src: NDArray,
    max_value=PIXEL_8BIT_MAX,
    method=ThresholdMethod.BINARY,
) -> ThresholdResult:
    computed_threshold_value, threshold_image = cv2.threshold(
        src, 0, max_value, method.value | cv2.THRESH_OTSU
    )
    return ThresholdResult(computed_threshold_value, threshold_image)


def threshold_triangle(
    src: NDArray,
    max_value=PIXEL_8BIT_MAX,
    method=ThresholdMethod.BINARY,
) -> ThresholdResult:
    computed_threshold_value, threshold_image = cv2.threshold(
        src, 0, max_value, method.value | cv2.THRESH_TRIANGLE
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
    """
    Constant subtracted from the mean or weighted mean (see the details below).
    Normally, it is positive but may be zero or negative as well.
    """

    assert method in ADAPTIVE_THRESHOLD_METHODS
    assert block_size >= 3
    assert block_size >= 3
    assert block_size % 2 == 1
    # Size of a pixel neighborhood
    # that is used to calculate a threshold value for the pixel

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
    def cvl_threshold_otsu(
        src: NDArray,
        max_value=PIXEL_8BIT_MAX,
        method=ThresholdMethod.BINARY,
    ):
        return threshold_otsu(src, max_value, method)

    @staticmethod
    def cvl_threshold_triangle(
        src: NDArray,
        max_value=PIXEL_8BIT_MAX,
        method=ThresholdMethod.BINARY,
    ):
        return threshold_triangle(src, max_value, method)

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

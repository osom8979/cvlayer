# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.adaptive_threshold import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_C,
    AdaptiveMethod,
    ThresholdMethod,
    adaptive_threshold,
)
from cvlayer.cv.color import PIXEL_8BIT_MAX


class CvlAdaptiveThreshold:
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

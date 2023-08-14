# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.color import PIXEL_8BIT_HALF, PIXEL_8BIT_MAX
from cvlayer.cv.threshold import ThresholdMethod, ThresholdResult, threshold


class CvlThreshold:
    ThresholdMethodType = ThresholdMethod
    ThresholdResultType = ThresholdResult

    @staticmethod
    def cvl_threshold(
        src: NDArray,
        thresh=PIXEL_8BIT_HALF,
        max_value=PIXEL_8BIT_MAX,
        method=ThresholdMethod.BINARY,
    ) -> ThresholdResult:
        return threshold(src, thresh, max_value, method)

# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

CANNY_THRESHOLD_MIN: Final[int] = 30
CANNY_THRESHOLD_MAX: Final[int] = 70


def canny(
    src: NDArray,
    threshold_min=CANNY_THRESHOLD_MIN,
    threshold_max=CANNY_THRESHOLD_MAX,
) -> NDArray:
    return cv2.Canny(src, threshold_min, threshold_max)


class CvlFilterEdgeCanny:
    @staticmethod
    def cvl_canny(
        src: NDArray,
        threshold_min=CANNY_THRESHOLD_MIN,
        threshold_max=CANNY_THRESHOLD_MAX,
    ):
        return canny(src, threshold_min, threshold_max)

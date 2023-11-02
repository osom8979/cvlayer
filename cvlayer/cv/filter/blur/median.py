# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

from cvlayer.typing import SizeI

DEFAULT_KSIZE: Final[SizeI] = (3, 3)


def median_blur(src: NDArray, ksize=DEFAULT_KSIZE) -> NDArray:
    return cv2.medianBlur(src, ksize)


class CvlFilterBlurMedian:
    @staticmethod
    def cvl_median_blur(src: NDArray, ksize=DEFAULT_KSIZE):
        return median_blur(src, ksize)

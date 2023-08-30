# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

CONVERT_SCALE_ABS_ALPHA: Final[float] = 1.0
CONVERT_SCALE_ABS_BERA: Final[float] = 0.0


def convert_scale_abs(
    src: NDArray,
    alpha=CONVERT_SCALE_ABS_ALPHA,
    beta=CONVERT_SCALE_ABS_BERA,
) -> NDArray:
    return cv2.convertScaleAbs(src, None, alpha=alpha, beta=beta)


class CvlConvertScaleAbs:
    @staticmethod
    def cvl_convert_scale_abs(
        src: NDArray,
        alpha=CONVERT_SCALE_ABS_ALPHA,
        beta=CONVERT_SCALE_ABS_BERA,
    ):
        return convert_scale_abs(src, alpha, beta)

# -*- coding: utf-8 -*-

from typing import Final, Optional

import cv2
from numpy import clip, uint8
from numpy.typing import NDArray

from cvlayer.typing import Number

PIXEL_8BIT_MIN: Final[int] = 0
PIXEL_8BIT_MAX: Final[int] = 255
PIXEL_8BIT_HALF: Final[int] = PIXEL_8BIT_MAX // 2


def saturate(
    src: NDArray,
    alpha=1.0,
    a_min: Number = PIXEL_8BIT_MIN,
    a_max: Number = PIXEL_8BIT_MAX,
    dtype: Optional[type] = None,
) -> NDArray:
    """
    dst(x, y) = saturate(src(x, y) + (src(x, y) - 128) * a)
    """
    a_half = a_min + abs(a_max - a_min) / 2
    dst = clip((1.0 + alpha) * src - a_half * alpha, a_min, a_max)
    if dtype is not None:
        return dst.astype(dtype)
    else:
        return dst


def convert_scale_abs(
    src: NDArray,
    alpha=1.0,
    beta=0.0,
) -> NDArray[uint8]:
    return cv2.convertScaleAbs(src, None, alpha, beta)


class CvlColor:
    @staticmethod
    def cvl_saturate(
        src: NDArray[uint8],
        alpha: float,
        a_min: Number = PIXEL_8BIT_MIN,
        a_max: Number = PIXEL_8BIT_MAX,
    ):
        return saturate(src, alpha, a_min, a_max)

    @staticmethod
    def cvl_convert_scale_abs(
        src: NDArray,
        alpha=1.0,
        beta=0.0,
    ) -> NDArray:
        return convert_scale_abs(src, alpha, beta)

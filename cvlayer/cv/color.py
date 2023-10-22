# -*- coding: utf-8 -*-

from typing import Final, Optional

import cv2
from numpy import clip, int32, uint8
from numpy.typing import NDArray

from cvlayer.palette import css4_palette
from cvlayer.typing import Color, ColorLike, Number

PIXEL_8BIT_MIN: Final[int] = 0
PIXEL_8BIT_MAX: Final[int] = 255
PIXEL_8BIT_HALF: Final[int] = PIXEL_8BIT_MAX // 2


def normalize_color(color: ColorLike) -> Color:
    if isinstance(color, (int, float)):
        return (color,)
    elif isinstance(color, str):
        return css4_palette()[color.upper()]
    elif isinstance(color, (tuple, list)):
        return tuple(c for c in color)
    else:
        raise TypeError(f"Unsupported color type: {type(color).__name__}")


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


def shift_degree_channel(
    src: NDArray,
    shift=0,
    maxval=180,
) -> NDArray:
    if 0 <= shift < 256 - 180:
        return (src + shift) % maxval
    else:
        return uint8((int32(src) + shift) % maxval)  # type: ignore[return-value]


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

    @staticmethod
    def cvl_shift_degree_channel(
        src: NDArray,
        shift: int,
        maxval=180,
    ):
        return shift_degree_channel(src, shift, maxval)

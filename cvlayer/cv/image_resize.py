# -*- coding: utf-8 -*-

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.interpolation import (
    DEFAULT_INTERPOLATION,
    normalize_interpolation,
)


def resize_constant(
    src: NDArray,
    x: int,
    y: int,
    interpolation=DEFAULT_INTERPOLATION,
) -> NDArray:
    _interpolation = normalize_interpolation(interpolation)
    return cv2.resize(src, dsize=(x, y), interpolation=_interpolation)


def resize_ratio(
    src: NDArray,
    x: float,
    y: float,
    interpolation=DEFAULT_INTERPOLATION,
) -> NDArray:
    _interpolation = normalize_interpolation(interpolation)
    return cv2.resize(src, dsize=(0, 0), fx=x, fy=y, interpolation=_interpolation)


class CvlImageResize:
    @staticmethod
    def cvl_resize_constant(
        src: NDArray,
        x: int,
        y: int,
        interpolation=DEFAULT_INTERPOLATION,
    ) -> NDArray:
        return resize_constant(src, x, y, interpolation)

    @staticmethod
    def cvl_resize_ratio(
        src: NDArray,
        x: float,
        y: float,
        interpolation=DEFAULT_INTERPOLATION,
    ) -> NDArray:
        return resize_ratio(src, x, y, interpolation)

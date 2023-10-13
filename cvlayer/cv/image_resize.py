# -*- coding: utf-8 -*-

import cv2
from numpy.typing import NDArray

from cvlayer.cv.interpolation import Interpolation


def resize_constant(
    src: NDArray,
    x: int,
    y: int,
    interpolation=Interpolation.INTER_NEAREST,
) -> NDArray:
    return cv2.resize(
        src,
        dsize=(x, y),
        interpolation=interpolation.value,
    )


def resize_ratio(
    src: NDArray,
    x: float,
    y: float,
    interpolation=Interpolation.INTER_NEAREST,
) -> NDArray:
    return cv2.resize(
        src,
        dsize=(0, 0),
        fx=x,
        fy=y,
        interpolation=interpolation.value,
    )


class CvlImageResize:
    @staticmethod
    def cvl_resize_constant(
        src: NDArray,
        x: int,
        y: int,
        interpolation=Interpolation.INTER_NEAREST,
    ) -> NDArray:
        return resize_constant(src, x, y, interpolation)

    @staticmethod
    def cvl_resize_ratio(
        src: NDArray,
        x: float,
        y: float,
        interpolation=Interpolation.INTER_NEAREST,
    ) -> NDArray:
        return resize_ratio(src, x, y, interpolation)

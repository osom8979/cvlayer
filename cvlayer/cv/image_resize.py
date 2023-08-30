# -*- coding: utf-8 -*-

from enum import Enum, unique

import cv2
from numpy.typing import NDArray


@unique
class Interpolation(Enum):
    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_AREA = cv2.INTER_AREA
    INTER_LANCZOS4 = cv2.INTER_LANCZOS4
    INTER_LINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    INTER_NEAREST_EXACT = cv2.INTER_NEAREST_EXACT
    INTER_MAX = cv2.INTER_MAX


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

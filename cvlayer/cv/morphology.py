# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final

import cv2
from numpy.typing import NDArray

from cvlayer.typing import PointInt, SizeInt

DEFAULT_KSIZE: Final[SizeInt] = (3, 3)
DEFAULT_ANCHOR: Final[PointInt] = (-1, -1)
DEFAULT_ITERATIONS: Final[int] = 1

DEFAULT_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    DEFAULT_KSIZE,
    DEFAULT_ANCHOR,
)


@unique
class MorphMethod(Enum):
    RECT = cv2.MORPH_RECT
    CROSS = cv2.MORPH_CROSS
    ELLIPSE = cv2.MORPH_ELLIPSE


def get_structuring_element(
    shape=MorphMethod.RECT,
    ksize=DEFAULT_KSIZE,
    anchor=DEFAULT_ANCHOR,
) -> NDArray:
    return cv2.getStructuringElement(shape.value, ksize, anchor)


def erode(
    src: NDArray,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
) -> NDArray:
    return cv2.erode(src, kernel, None, anchor, iterations)


def dilate(
    src: NDArray,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
) -> NDArray:
    return cv2.dilate(src, kernel, None, anchor, iterations)


class CvlMorphology:
    @staticmethod
    def cvl_get_structuring_element(
        shape=MorphMethod.RECT,
        ksize=DEFAULT_KSIZE,
        anchor=DEFAULT_ANCHOR,
    ):
        return get_structuring_element(shape, ksize, anchor)

    @staticmethod
    def cvl_erode(
        src: NDArray,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
    ):
        return erode(src, kernel, anchor, iterations)

    @staticmethod
    def cvl_dilate(
        src: NDArray,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
    ):
        return dilate(src, kernel, anchor, iterations)

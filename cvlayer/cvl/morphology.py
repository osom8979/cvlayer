# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.morphology import (
    DEFAULT_ANCHOR,
    DEFAULT_ITERATIONS,
    DEFAULT_KERNEL,
    DEFAULT_KSIZE,
    MorphMethod,
    dilate,
    erode,
    get_structuring_element,
)


class CvlMorphology:
    MorphMethodType = MorphMethod

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

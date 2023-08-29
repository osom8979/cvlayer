# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.resize import Interpolation, resize_constant, resize_ratio


class CvlResize:
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

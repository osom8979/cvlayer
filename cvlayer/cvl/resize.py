# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.resize import DEFAULT_INTERPOLATION, Interpolation, resize


class CvlResize:
    InterpolationType = Interpolation

    @staticmethod
    def cvl_resize(
        src: NDArray,
        scale: float,
        interpolation=DEFAULT_INTERPOLATION,
    ) -> NDArray:
        return resize(src, scale, interpolation)

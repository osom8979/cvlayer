# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.in_range import in_range, in_range2


class CvlInRange:
    @staticmethod
    def cvl_in_range(
        src: NDArray,
        lower_bound: NDArray,
        upper_bound: NDArray,
    ) -> NDArray:
        return in_range(src, lower_bound, upper_bound)

    @staticmethod
    def cvl_in_range2(
        src: NDArray,
        lower_bound1: NDArray,
        upper_bound1: NDArray,
        lower_bound2: NDArray,
        upper_bound2: NDArray,
    ) -> NDArray:
        return in_range2(src, lower_bound1, upper_bound1, lower_bound2, upper_bound2)

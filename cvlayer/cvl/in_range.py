# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.in_range import in_range


class CvlInRange:
    @staticmethod
    def cvl_in_range(src: NDArray, lower_bound, upper_bound) -> NDArray:
        return in_range(src, lower_bound, upper_bound)

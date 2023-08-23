# -*- coding: utf-8 -*-

import cv2
from numpy.typing import NDArray


def in_range(src: NDArray, lower_bound: NDArray, upper_bound: NDArray) -> NDArray:
    return cv2.inRange(src, lower_bound, upper_bound)


def in_range2(
    src: NDArray,
    lower_bound1: NDArray,
    upper_bound1: NDArray,
    lower_bound2: NDArray,
    upper_bound2: NDArray,
) -> NDArray:
    mask1 = cv2.inRange(src, lower_bound1, upper_bound1)
    mask2 = cv2.inRange(src, lower_bound2, upper_bound2)
    return cv2.bitwise_or(mask1, mask2)

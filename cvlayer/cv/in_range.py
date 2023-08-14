# -*- coding: utf-8 -*-

import cv2
from numpy.typing import NDArray


def in_range(src: NDArray, lower_bound, upper_bound) -> NDArray:
    return cv2.inRange(src, lower_bound, upper_bound)

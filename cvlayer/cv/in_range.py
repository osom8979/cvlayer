# -*- coding: utf-8 -*-

from typing import Sequence, Union

import cv2
from numpy.typing import NDArray

from cvlayer.typing import NumberT


def in_range(
    src: NDArray,
    lower_bound: Union[NDArray, Sequence[NumberT]],
    upper_bound: Union[NDArray, Sequence[NumberT]],
) -> NDArray:
    return cv2.inRange(src, lower_bound, upper_bound)  # type: ignore[arg-type]


def in_range2(
    src: NDArray,
    lower_bound1: Union[NDArray, Sequence[NumberT]],
    upper_bound1: Union[NDArray, Sequence[NumberT]],
    lower_bound2: Union[NDArray, Sequence[NumberT]],
    upper_bound2: Union[NDArray, Sequence[NumberT]],
) -> NDArray:
    mask1 = in_range(src, lower_bound1, upper_bound1)
    mask2 = in_range(src, lower_bound2, upper_bound2)
    return cv2.bitwise_or(mask1, mask2)


class CvlInRange:
    @staticmethod
    def cvl_in_range(
        src: NDArray,
        lower_bound: Union[NDArray, Sequence[NumberT]],
        upper_bound: Union[NDArray, Sequence[NumberT]],
    ):
        return in_range(src, lower_bound, upper_bound)

    @staticmethod
    def cvl_in_range2(
        src: NDArray,
        lower_bound1: Union[NDArray, Sequence[NumberT]],
        upper_bound1: Union[NDArray, Sequence[NumberT]],
        lower_bound2: Union[NDArray, Sequence[NumberT]],
        upper_bound2: Union[NDArray, Sequence[NumberT]],
    ):
        return in_range2(src, lower_bound1, upper_bound1, lower_bound2, upper_bound2)

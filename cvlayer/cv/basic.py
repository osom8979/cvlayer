# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import cv2
from numpy.typing import NDArray


def mean(
    src: NDArray,
    mask: Optional[NDArray] = None,
) -> Tuple[float, float, float, float]:
    means = cv2.mean(src, mask)
    assert len(means) == 4
    return means[0], means[1], means[2], means[3]


class CvlBasic:
    @staticmethod
    def cvl_mean(src: NDArray, mask: Optional[NDArray] = None):
        return mean(src, mask)

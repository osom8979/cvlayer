# -*- coding: utf-8 -*-

import cv2
from numpy import float32, ndarray
from numpy.typing import NDArray


def move(src: NDArray, x: int, y: int) -> NDArray:
    # fmt: off
    m = [[1, 0, x],
         [0, 1, y]]
    # fmt: on

    fm = float32(m)  # type: ignore[arg-type]
    assert isinstance(fm, ndarray)
    assert fm.dtype.name == "float32"
    height = src.shape[0]
    width = src.shape[1]
    size = width, height

    return cv2.warpAffine(src, fm, size)


class CvlImageMove:
    @staticmethod
    def cvl_move(src: NDArray, x: int, y: int):
        return move(src, x, y)

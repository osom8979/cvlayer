# -*- coding: utf-8 -*-

from typing import Final, Tuple

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.border import DEFAULT_BORDER_TYPE, normalize_border_type
from cvlayer.cv.types.ddepth import (
    DEFAULT_DESIRED_DEPTH,
    normalize_desired_depth,
    validate_depth_combinations,
)

DEFAULT_DX: Final[int] = 1
DEFAULT_DY: Final[int] = 1
DEFAULT_KERNEL_SIZE: Final[int] = 3
DEFAULT_SCALE: Final[float] = 1.0
DEFAULT_DELTA: Final[float] = 0.0
AVAILABLE_KERNEL_SIZE: Tuple[int, int, int, int] = 1, 3, 5, 7


def sobel(
    frame: NDArray,
    ddepth=DEFAULT_DESIRED_DEPTH,
    dx=DEFAULT_DX,
    dy=DEFAULT_DY,
    kernel_size=DEFAULT_KERNEL_SIZE,
    scale=DEFAULT_SCALE,
    delta=DEFAULT_DELTA,
    border=DEFAULT_BORDER_TYPE,
):
    assert dx or dy
    assert kernel_size % 2 == 1
    assert kernel_size >= 1
    assert kernel_size in AVAILABLE_KERNEL_SIZE

    _ddepth = normalize_desired_depth(ddepth)
    validate_depth_combinations(frame, _ddepth)
    _border = normalize_border_type(border)
    if _border == cv2.BORDER_WRAP:
        raise ValueError("Unsupported border type: BORDER_WRAP")

    return cv2.Sobel(
        frame,
        _ddepth,
        dx,
        dy,
        None,
        kernel_size,
        scale,
        delta,
        _border,
    )


class CvlSobel:
    @staticmethod
    def cvl_sobel(
        frame: NDArray,
        ddepth=DEFAULT_DESIRED_DEPTH,
        dx=DEFAULT_DX,
        dy=DEFAULT_DY,
        kernel_size=DEFAULT_KERNEL_SIZE,
        scale=DEFAULT_SCALE,
        delta=DEFAULT_DELTA,
        border=DEFAULT_BORDER_TYPE,
    ):
        return sobel(
            frame,
            ddepth,
            dx,
            dy,
            kernel_size,
            scale,
            delta,
            border,
        )

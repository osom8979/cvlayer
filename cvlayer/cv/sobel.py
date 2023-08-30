# -*- coding: utf-8 -*-

from typing import Final, Tuple

import cv2
from numpy.typing import NDArray

from cvlayer.cv.border import DEFAULT_BORDER_TYPE
from cvlayer.cv.depth import DEFAULT_OUTPUT_DEPTH, validate_depth_combinations

DEFAULT_DX: Final[int] = 1
DEFAULT_DY: Final[int] = 1
DEFAULT_KERNEL_SIZE: Final[int] = 1
DEFAULT_SCALE: Final[float] = 1.0
DEFAULT_DELTA: Final[float] = 0.0
AVAILABLE_KERNEL_SIZE: Tuple[int, int, int, int] = 1, 3, 5, 7


def sobel(
    frame: NDArray,
    output_depth=DEFAULT_OUTPUT_DEPTH,
    dx=DEFAULT_DX,
    dy=DEFAULT_DY,
    kernel_size=DEFAULT_KERNEL_SIZE,
    scale=DEFAULT_SCALE,
    delta=DEFAULT_DELTA,
    border_type=DEFAULT_BORDER_TYPE,
):
    assert dx or dy
    assert kernel_size % 2 == 1
    assert kernel_size >= 1
    assert kernel_size in AVAILABLE_KERNEL_SIZE

    validate_depth_combinations(frame, output_depth.value)

    if border_type.value == cv2.BORDER_WRAP:
        raise ValueError("Unsupported border type: BORDER_WRAP")

    return cv2.Sobel(
        frame,
        output_depth.value,
        dx,
        dy,
        None,
        kernel_size,
        scale,
        delta,
        border_type.value,
    )


class CvlSobel:
    @staticmethod
    def cvl_sobel(
        frame: NDArray,
        output_depth=DEFAULT_OUTPUT_DEPTH,
        dx=DEFAULT_DX,
        dy=DEFAULT_DY,
        kernel_size=DEFAULT_KERNEL_SIZE,
        scale=DEFAULT_SCALE,
        delta=DEFAULT_DELTA,
        border_type=DEFAULT_BORDER_TYPE,
    ):
        return sobel(
            frame,
            output_depth,
            dx,
            dy,
            kernel_size,
            scale,
            delta,
            border_type,
        )

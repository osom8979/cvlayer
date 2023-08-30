# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final

import cv2
from numpy.typing import NDArray

from cvlayer.cv.data_type import (
    CV_16S,
    CV_32F,
    CV_64F,
    SAME_DEPTH_AS_SOURCE,
    validate_depth_combinations,
)


@unique
class LaplacianOutputDepth(Enum):
    SAME_INPUT = SAME_DEPTH_AS_SOURCE
    INT16 = CV_16S
    FLOAT32 = CV_32F
    FLOAT64 = CV_64F


@unique
class LaplacianBorder(Enum):
    CONSTANT = cv2.BORDER_CONSTANT
    REPLICATE = cv2.BORDER_REPLICATE
    REFLECT = cv2.BORDER_REFLECT
    # BORDER = cv2.BORDER_WRAP  # BORDER_WRAP is not supported
    REFLECT101 = cv2.BORDER_REFLECT101
    TRANSPARENT = cv2.BORDER_TRANSPARENT
    ISOLATED = cv2.BORDER_ISOLATED


DEFAULT_OUTPUT_DEPTH: Final[LaplacianOutputDepth] = LaplacianOutputDepth.SAME_INPUT

DEFAULT_KERNEL_SIZE: Final[int] = 1
"""
Aperture size used to compute the second-derivative filters.
See getDerivKernels for details.
The size must be positive and odd.
"""

DEFAULT_SCALE: Final[float] = 1.0
"""
scale factor for the computed Laplacian values.
By default, no scaling is applied.
See getDerivKernels for details.
"""

DEFAULT_DELTA: Final[float] = 0.0
"""delta value that is added to the results prior to storing them in dst."""

DEFAULT_BORDER_TYPE: Final[LaplacianBorder] = LaplacianBorder.REFLECT101

CONVERT_SCALE_ABS_ALPHA: Final[float] = 1.0
CONVERT_SCALE_ABS_BERA: Final[float] = 0.0


def laplacian(
    src: NDArray,
    output_depth=DEFAULT_OUTPUT_DEPTH,
    kernel_size=DEFAULT_KERNEL_SIZE,
    scale=DEFAULT_SCALE,
    delta=DEFAULT_DELTA,
    border_type=DEFAULT_BORDER_TYPE,
) -> NDArray:
    validate_depth_combinations(src.dtype, output_depth.value)

    assert kernel_size % 2 == 1
    assert kernel_size >= 1

    if border_type.value == cv2.BORDER_WRAP:
        raise ValueError("Unsupported border type: BORDER_WRAP")

    return cv2.Laplacian(
        src=src,
        ddepth=output_depth.value,
        dst=None,
        ksize=kernel_size,
        scale=scale,
        delta=delta,
        borderType=border_type.value,
    )


def convert_scale_abs(
    src: NDArray,
    alpha=CONVERT_SCALE_ABS_ALPHA,
    beta=CONVERT_SCALE_ABS_BERA,
) -> NDArray:
    return cv2.convertScaleAbs(src, None, alpha=alpha, beta=beta)


class CvlLaplacian:
    @staticmethod
    def cvl_laplacian(
        src: NDArray,
        output_depth=DEFAULT_OUTPUT_DEPTH,
        kernel_size=DEFAULT_KERNEL_SIZE,
        scale=DEFAULT_SCALE,
        delta=DEFAULT_DELTA,
        border_type=DEFAULT_BORDER_TYPE,
    ):
        return laplacian(src, output_depth, kernel_size, scale, delta, border_type)

    @staticmethod
    def cvl_convert_scale_abs(
        src: NDArray,
        alpha=CONVERT_SCALE_ABS_ALPHA,
        beta=CONVERT_SCALE_ABS_BERA,
    ):
        return convert_scale_abs(src, alpha, beta)

# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.border import DEFAULT_BORDER_TYPE, normalize_border_type
from cvlayer.cv.types.ddepth import (
    DEFAULT_DESIRED_DEPTH,
    normalize_desired_depth,
    validate_depth_combinations,
)

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


def laplacian(
    src: NDArray,
    ddepth=DEFAULT_DESIRED_DEPTH,
    kernel_size=DEFAULT_KERNEL_SIZE,
    scale=DEFAULT_SCALE,
    delta=DEFAULT_DELTA,
    border=DEFAULT_BORDER_TYPE,
) -> NDArray:
    assert kernel_size % 2 == 1
    assert kernel_size >= 1

    _ddepth = normalize_desired_depth(ddepth)
    validate_depth_combinations(src, _ddepth)

    _border = normalize_border_type(border)

    if _border == cv2.BORDER_WRAP:
        raise ValueError("Unsupported border type: BORDER_WRAP")

    return cv2.Laplacian(src, _ddepth, None, kernel_size, scale, delta, _border)


class CvlFilterEdgeLaplacian:
    @staticmethod
    def cvl_laplacian(
        src: NDArray,
        ddepth=DEFAULT_DESIRED_DEPTH,
        kernel_size=DEFAULT_KERNEL_SIZE,
        scale=DEFAULT_SCALE,
        delta=DEFAULT_DELTA,
        border=DEFAULT_BORDER_TYPE,
    ):
        return laplacian(src, ddepth, kernel_size, scale, delta, border)

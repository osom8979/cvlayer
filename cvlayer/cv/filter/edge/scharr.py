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

DEFAULT_DX: Final[int] = 1
"""order of the derivative x"""

DEFAULT_DY: Final[int] = 1
"""order of the derivative y"""

DEFAULT_SCALE: Final[float] = 1.0
"""
optional scale factor for the computed derivative values;
by default, no scaling is applied
(see getDerivKernels for details)
"""

DEFAULT_DELTA: Final[float] = 0.0
"""
optional delta value that is added to the results prior to storing them in dst.
"""


def scharr(
    src: NDArray,
    ddepth=DEFAULT_DESIRED_DEPTH,
    dx=DEFAULT_DX,
    dy=DEFAULT_DY,
    scale=DEFAULT_SCALE,
    delta=DEFAULT_DELTA,
    border=DEFAULT_BORDER_TYPE,
) -> NDArray:
    _ddepth = normalize_desired_depth(ddepth)
    validate_depth_combinations(src, _ddepth)

    _border = normalize_border_type(border)

    if _border == cv2.BORDER_WRAP:
        raise ValueError("Unsupported border type: BORDER_WRAP")

    return cv2.Scharr(src, _ddepth, dx, dy, None, scale, delta, _border)


class CvlFilterEdgeScharr:
    @staticmethod
    def cvl_scharr(
        src: NDArray,
        ddepth=DEFAULT_DESIRED_DEPTH,
        dx=DEFAULT_DX,
        dy=DEFAULT_DY,
        scale=DEFAULT_SCALE,
        delta=DEFAULT_DELTA,
        border=DEFAULT_BORDER_TYPE,
    ):
        return scharr(src, ddepth, dx, dy, scale, delta, border)

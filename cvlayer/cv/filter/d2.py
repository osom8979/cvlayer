# -*- coding: utf-8 -*-

from typing import Optional

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.border import DEFAULT_BORDER_TYPE, normalize_border_type
from cvlayer.cv.types.ddepth import (
    DEFAULT_DESIRED_DEPTH,
    normalize_desired_depth,
    validate_depth_combinations,
)
from cvlayer.typing import PointI


def filter_2d(
    src: NDArray,
    kernel: NDArray,
    ddepth=DEFAULT_DESIRED_DEPTH,
    anchor: Optional[PointI] = None,
    delta=0.0,
    border=DEFAULT_BORDER_TYPE,
) -> NDArray:
    _ddepth = normalize_desired_depth(ddepth)
    validate_depth_combinations(src, _ddepth)
    _anchor = anchor if anchor is not None else (-1, -1)
    _border = normalize_border_type(border)
    if _border == cv2.BORDER_WRAP:
        raise ValueError("BORDER_WRAP is not supported")
    return cv2.filter2D(src, _ddepth, kernel, None, _anchor, delta, _border)


class CvlFilterD2:
    @staticmethod
    def cvl_filter_2d(
        src: NDArray,
        kernel: NDArray,
        ddepth=DEFAULT_DESIRED_DEPTH,
        anchor: Optional[PointI] = None,
        delta=0.0,
        border=DEFAULT_BORDER_TYPE,
    ):
        return filter_2d(src, kernel, ddepth, anchor, delta, border)

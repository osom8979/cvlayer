# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

CV_8U: Final[int] = cv2.CV_8U  # type: ignore[attr-defined]
CANNY_THRESHOLD_MIN: Final[int] = 30
CANNY_THRESHOLD_MAX: Final[int] = 70


def edge_detector_canny(
    src: NDArray,
    threshold_min=CANNY_THRESHOLD_MIN,
    threshold_max=CANNY_THRESHOLD_MAX,
) -> NDArray:
    return cv2.Canny(src, threshold_min, threshold_max)


def edge_detector_laplacian(src: NDArray, ddepth=CV_8U) -> NDArray:
    return cv2.Laplacian(src, ddepth)

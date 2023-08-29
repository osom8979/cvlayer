# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.edge_detector import (
    CANNY_THRESHOLD_MAX,
    CANNY_THRESHOLD_MIN,
    CV_8U,
    canny,
    laplacian,
)


class CvlEdgeDetector:
    @staticmethod
    def cvl_canny(
        src: NDArray,
        threshold_min=CANNY_THRESHOLD_MIN,
        threshold_max=CANNY_THRESHOLD_MAX,
    ) -> NDArray:
        return canny(src, threshold_min, threshold_max)

    @staticmethod
    def cvl_laplacian(src: NDArray, ddepth=CV_8U) -> NDArray:
        return laplacian(src, ddepth)

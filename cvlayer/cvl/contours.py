# -*- coding: utf-8 -*-

import cv2
from numpy.typing import NDArray

from cvlayer.cv.contours import convert_roi2contour, find_largest_contour_index
from cvlayer.types import Rect


class CvlContours:
    @staticmethod
    def cvl_contour_area(contour: NDArray, oriented=False) -> float:
        return cv2.contourArea(contour, oriented)

    @staticmethod
    def cvl_find_largest_contour_index(contour: NDArray, oriented=False) -> int:
        return find_largest_contour_index(contour, oriented)

    @staticmethod
    def cvl_convert_roi2contour(roi: Rect) -> NDArray:
        return convert_roi2contour(roi)

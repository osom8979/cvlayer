# -*- coding: utf-8 -*-

from typing import List

from numpy.typing import NDArray

from cvlayer.cv.contours_edge import (
    FindContourEdgeMethod,
    find_best_contour_edge_points,
)


class CvlContoursEdge:
    @staticmethod
    def find_leftmost_contour(contours: List[NDArray]):
        return find_best_contour_edge_points(FindContourEdgeMethod.LEFT, contours)

    @staticmethod
    def find_rightmost_contour(contours: List[NDArray]):
        return find_best_contour_edge_points(FindContourEdgeMethod.RIGHT, contours)

    @staticmethod
    def find_topmost_contour(contours: List[NDArray]):
        return find_best_contour_edge_points(FindContourEdgeMethod.TOP, contours)

    @staticmethod
    def find_bottommost_contour(contours: List[NDArray]):
        return find_best_contour_edge_points(FindContourEdgeMethod.BOTTOM, contours)

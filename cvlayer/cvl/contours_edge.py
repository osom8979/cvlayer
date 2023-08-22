# -*- coding: utf-8 -*-

from typing import List

from numpy.typing import NDArray

from cvlayer.cv.contours_edge import (
    FindContourEdgeMethod,
    find_best_contour_edge_points,
)


class CvlContoursEdge:
    @staticmethod
    def cvl_find_leftmost_contour(contours: List[NDArray]):
        return find_best_contour_edge_points(FindContourEdgeMethod.LEFT, contours)

    @staticmethod
    def cvl_find_rightmost_contour(contours: List[NDArray]):
        return find_best_contour_edge_points(FindContourEdgeMethod.RIGHT, contours)

    @staticmethod
    def cvl_find_topmost_contour(contours: List[NDArray]):
        return find_best_contour_edge_points(FindContourEdgeMethod.TOP, contours)

    @staticmethod
    def cvl_find_bottommost_contour(contours: List[NDArray]):
        return find_best_contour_edge_points(FindContourEdgeMethod.BOTTOM, contours)

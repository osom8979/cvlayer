# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.contours import (
    MinAreaRectResult,
    approx_poly_dp,
    arc_length,
    box_points,
    contour_area,
    convert_roi2contour,
    convex_hull,
    find_largest_contour_index,
    min_area_rect,
)
from cvlayer.types import Rect


class CvlContours:
    MinAreaRectResultType = MinAreaRectResult

    @staticmethod
    def cvl_find_largest_contour_index(contour: NDArray, oriented=False) -> int:
        return find_largest_contour_index(contour, oriented)

    @staticmethod
    def cvl_convert_roi2contour(roi: Rect) -> NDArray:
        return convert_roi2contour(roi)

    @staticmethod
    def cvl_contour_area(contour: NDArray, oriented=False) -> float:
        return contour_area(contour, oriented)

    @staticmethod
    def cvl_convex_hull(contour: NDArray) -> NDArray:
        return convex_hull(contour)

    @staticmethod
    def cvl_arc_length(curve: NDArray, closed=False) -> float:
        return arc_length(curve, closed)

    @staticmethod
    def cvl_approx_poly_dp(curve: NDArray, epsilon: float, closed=False) -> NDArray:
        return approx_poly_dp(curve, epsilon, closed)

    @staticmethod
    def cvl_min_area_rect(points: NDArray) -> MinAreaRectResult:
        return min_area_rect(points)

    @staticmethod
    def cvl_box_points(box: MinAreaRectResult) -> NDArray:
        return box_points(box)

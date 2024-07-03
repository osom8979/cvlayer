# -*- coding: utf-8 -*-

from math import floor
from typing import NamedTuple

import cv2
from cv2.typing import MatLike
from numpy import logical_and, zeros
from numpy.typing import NDArray

from cvlayer.typing import PointF, RectI, SizeI


class RotatedRect(NamedTuple):
    center: PointF
    size: SizeI
    rotation: float


def min_area_rect(points: NDArray) -> RotatedRect:
    center, size, rotation = cv2.minAreaRect(points)
    cx, cy = center
    w, h = size
    return RotatedRect((cx, cy), (floor(w), floor(h)), rotation)


def box_points(box: RotatedRect) -> NDArray:
    center, size, rotation = box
    return cv2.boxPoints((center, size, rotation))


def bitwise_intersection_contours(
    width: int, height: int, contour1: MatLike, contour2: MatLike
) -> NDArray:
    blank1 = zeros((height, width))
    blank2 = zeros((height, width))

    mask1 = cv2.drawContours(blank1, [contour1], 0, [1])
    mask2 = cv2.drawContours(blank2, [contour2], 1, [1])

    return logical_and(mask1, mask2)


def contour_area(contour: NDArray, oriented=False) -> float:
    return cv2.contourArea(contour, oriented)


def convex_hull(contour: NDArray) -> NDArray:
    return cv2.convexHull(contour)


def arc_length(curve: NDArray, closed=False) -> float:
    return cv2.arcLength(curve, closed=closed)


def approx_poly_dp(curve: NDArray, epsilon: float, closed=False) -> NDArray:
    return cv2.approxPolyDP(curve, epsilon=epsilon, closed=closed)


def bounding_rect(array: NDArray) -> RectI:
    x, y, w, h = cv2.boundingRect(array)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return x1, y1, x2, y2


class CvlContourAnalysis:
    @staticmethod
    def cvl_min_area_rect(points: NDArray):
        return min_area_rect(points)

    @staticmethod
    def cvl_box_points(box: RotatedRect):
        return box_points(box)

    @staticmethod
    def cvl_bitwise_intersection_contours(
        width: int,
        height: int,
        contour1: NDArray,
        contour2: NDArray,
    ):
        return bitwise_intersection_contours(width, height, contour1, contour2)

    @staticmethod
    def cvl_contour_area(contour: NDArray, oriented=False):
        return contour_area(contour, oriented)

    @staticmethod
    def cvl_convex_hull(contour: NDArray):
        return convex_hull(contour)

    @staticmethod
    def cvl_arc_length(curve: NDArray, closed=False):
        return arc_length(curve, closed)

    @staticmethod
    def cvl_approx_poly_dp(curve: NDArray, epsilon: float, closed=False):
        return approx_poly_dp(curve, epsilon, closed)

    @staticmethod
    def cvl_bounding_rect(array: NDArray):
        return bounding_rect(array)

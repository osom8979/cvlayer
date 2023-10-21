# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Iterable, NamedTuple, Sequence

import cv2
from numpy import int32, logical_and, uint8, zeros
from numpy.typing import NDArray

from cvlayer.typing import Image, PointF, SizeF


@unique
class FindContoursMode(Enum):
    CCOMP = cv2.RETR_CCOMP
    EXTERNAL = cv2.RETR_EXTERNAL
    LIST = cv2.RETR_LIST
    TREE = cv2.RETR_TREE
    FLOODFILL = cv2.RETR_FLOODFILL


@unique
class FindContoursMethod(Enum):
    NONE = cv2.CHAIN_APPROX_NONE
    SIMPLE = cv2.CHAIN_APPROX_SIMPLE
    TC89_KCOS = cv2.CHAIN_APPROX_TC89_KCOS
    TC89_L1 = cv2.CHAIN_APPROX_TC89_L1


class RotatedRect(NamedTuple):
    center: PointF
    size: SizeF
    rotation: float


class FindContoursResult(NamedTuple):
    contours: Sequence[NDArray[int32]]
    hierarchy: NDArray[int32]


def find_contours(
    image: Image,
    mode=FindContoursMode.TREE,
    method=FindContoursMethod.SIMPLE,
) -> FindContoursResult:
    if mode != FindContoursMode.FLOODFILL:
        if image.dtype != uint8:
            raise ValueError("Only uint8 is supported as image.dtype")
        if len(image.shape) != 2:
            raise ValueError("Only single-channel is supported")
    contours, hierarchy = cv2.findContours(image, mode.value, method.value)
    return FindContoursResult(contours, hierarchy)


def find_largest_contour_index(contours: Iterable[NDArray], oriented=False) -> int:
    areas = list(map(lambda c: cv2.contourArea(c, oriented), contours))
    return areas.index(max(areas))


def bitwise_intersection_contours(
    width: int, height: int, contour1: NDArray, contour2: NDArray
) -> NDArray:
    blank1 = zeros(shape=(height, width))
    blank2 = zeros(shape=(height, width))

    mask1 = cv2.drawContours(blank1, [contour1], 0, 1)  # noqa
    mask2 = cv2.drawContours(blank2, [contour2], 1, 1)  # noqa

    return logical_and(mask1, mask2)


def contour_area(contour: NDArray, oriented=False) -> float:
    return cv2.contourArea(contour, oriented)


def convex_hull(contour: NDArray) -> NDArray:
    return cv2.convexHull(contour)


def arc_length(curve: NDArray, closed=False) -> float:
    return cv2.arcLength(curve, closed=closed)


def approx_poly_dp(curve: NDArray, epsilon: float, closed=False) -> NDArray:
    return cv2.approxPolyDP(curve, epsilon=epsilon, closed=closed)


def min_area_rect(points: NDArray) -> RotatedRect:
    center, size, rotation = cv2.minAreaRect(points)
    cx, cy = center
    w, h = size
    return RotatedRect((cx, cy), (w, h), rotation)


def box_points(box: RotatedRect) -> NDArray:
    return cv2.boxPoints(box)


class CvlContours:
    @staticmethod
    def cvl_find_contours(
        image: Image,
        mode=FindContoursMode.TREE,
        method=FindContoursMethod.SIMPLE,
    ):
        return find_contours(image, mode, method)

    @staticmethod
    def cvl_find_largest_contour_index(contours: Iterable[NDArray], oriented=False):
        return find_largest_contour_index(contours, oriented)

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
    def cvl_min_area_rect(points: NDArray):
        return min_area_rect(points)

    @staticmethod
    def cvl_box_points(box: RotatedRect):
        return box_points(box)

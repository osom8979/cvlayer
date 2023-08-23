# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Iterable, NamedTuple

import cv2
from numpy import logical_and, zeros
from numpy.typing import NDArray

from cvlayer.types import Image, PointFloat, SizeFloat


@unique
class FindContoursMode(Enum):
    CCOMP = cv2.RETR_CCOMP
    EXTERNAL = cv2.RETR_EXTERNAL
    LIST = cv2.RETR_LIST
    TREE = cv2.RETR_TREE
    # FLOODFILL = cv2.RETR_FLOODFILL


@unique
class FindContoursMethod(Enum):
    NONE = cv2.CHAIN_APPROX_NONE
    SIMPLE = cv2.CHAIN_APPROX_SIMPLE
    TC89_KCOS = cv2.CHAIN_APPROX_TC89_KCOS
    TC89_L1 = cv2.CHAIN_APPROX_TC89_L1


class MinAreaRectResult(NamedTuple):
    center: PointFloat
    size: SizeFloat
    rotation: float


def find_contours(
    image: Image,
    mode=FindContoursMode.TREE,
    method=FindContoursMethod.SIMPLE,
):
    return cv2.findContours(image, mode.value, method.value)


def find_largest_contour_index(contours: Iterable[NDArray], oriented=False) -> int:
    areas = list(map(lambda c: cv2.contourArea(c, oriented), contours))
    return areas.index(max(areas))


def bitwise_intersection_contours(
    width: int, height: int, contour1: NDArray, contour2: NDArray
) -> NDArray:
    blank1 = zeros(shape=(height, width))
    blank2 = blank1.copy()

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


def min_area_rect(points: NDArray) -> MinAreaRectResult:
    center, size, rotation = cv2.minAreaRect(points)
    cx, cy = center
    w, h = size
    return MinAreaRectResult((cx, cy), (w, h), rotation)


def box_points(box: MinAreaRectResult) -> NDArray:
    return cv2.boxPoints(box)

# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Iterable, NamedTuple

import cv2
from numpy import array, logical_and, ndarray, zeros
from numpy.typing import NDArray

from cvlayer.types import PointFloat, Rect, SizeFloat


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


def find_largest_contour_index(contours: Iterable[NDArray], oriented=False) -> int:
    areas = list(map(lambda c: cv2.contourArea(c, oriented), contours))
    return areas.index(max(areas))


def convert_roi2contour(roi: Rect) -> NDArray:
    x1, y1, x2, y2 = [v for v in roi]
    shell = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    raw = [p for xy in shell for p in xy]
    contour = array(raw).reshape((5, 1, 2))

    assert isinstance(contour, ndarray)
    assert len(contour.shape) == 3
    assert contour.shape[0] == 5
    assert contour.shape[1] == 1
    assert contour.shape[2] == 2

    return contour


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

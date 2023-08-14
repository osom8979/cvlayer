# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Iterable

import cv2
from numpy import array, logical_and, ndarray, zeros
from numpy.typing import NDArray

from cvlayer.types import Rect


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

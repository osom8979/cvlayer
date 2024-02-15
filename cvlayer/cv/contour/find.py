# -*- coding: utf-8 -*-

from typing import Iterable, NamedTuple, Sequence

import cv2
from numpy import int32, uint8
from numpy.typing import NDArray

from cvlayer.cv.types.chain_approx import DEFAULT_CHAIN_APPROX, normalize_chain_approx
from cvlayer.cv.types.retrieval import (
    DEFAULT_RETRIEVAL,
    RETR_FLOODFILL,
    normalize_retrieval,
)


class FindContoursResult(NamedTuple):
    contours: Sequence[NDArray[int32]]
    hierarchy: NDArray[int32]


def find_contours(
    image: NDArray,
    mode=DEFAULT_RETRIEVAL,
    method=DEFAULT_CHAIN_APPROX,
) -> FindContoursResult:
    _mode = normalize_retrieval(mode)
    _method = normalize_chain_approx(method)

    if _mode != RETR_FLOODFILL:
        if image.dtype != uint8:
            raise ValueError("Only uint8 is supported as image.dtype")
        if len(image.shape) != 2:
            raise ValueError("Only single-channel is supported")

    contours, hierarchy = cv2.findContours(image, _mode, _method)
    return FindContoursResult(contours, hierarchy)  # type: ignore[arg-type]


def find_largest_contour_index(contours: Iterable[NDArray], oriented=False) -> int:
    areas = list(map(lambda c: cv2.contourArea(c, oriented), contours))
    return areas.index(max(areas))


class CvlContourFind:
    @staticmethod
    def cvl_find_contours(
        image: NDArray,
        mode=DEFAULT_RETRIEVAL,
        method=DEFAULT_CHAIN_APPROX,
    ):
        return find_contours(image, mode, method)

    @staticmethod
    def cvl_find_largest_contour_index(contours: Iterable[NDArray], oriented=False):
        return find_largest_contour_index(contours, oriented)

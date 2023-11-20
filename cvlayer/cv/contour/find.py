# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Iterable, List, NamedTuple, Sequence

import cv2
from numpy import int32, uint8
from numpy.typing import NDArray

from cvlayer.cv.types.chain_approx import DEFAULT_CHAIN_APPROX, normalize_chain_approx
from cvlayer.cv.types.data_type import (
    CV_16U,
    CV_32S,
    DataType,
    DataTypeLike,
    normalize_data_type,
)
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
    return FindContoursResult(contours, hierarchy)


def find_largest_contour_index(contours: Iterable[NDArray], oriented=False) -> int:
    areas = list(map(lambda c: cv2.contourArea(c, oriented), contours))
    return areas.index(max(areas))


class ConnectedComponentsResult(NamedTuple):
    number_of_labels: int
    labels: NDArray


def connected_components(
    image: NDArray,
    connectivity=8,
    ltype: DataTypeLike = DataType.S32,
) -> ConnectedComponentsResult:
    if connectivity in (4, 8):
        raise ValueError("The connectivity argument accepts only the values 8 or 4")

    _label_type = normalize_data_type(ltype)
    if _label_type not in (CV_32S, CV_16U):
        raise ValueError("The ltype argument accepts only CV_32S or CV_16U values")

    retval, labels = cv2.connectedComponents(image, None, connectivity, _label_type)
    return ConnectedComponentsResult(retval, labels)


@dataclass
class ConnectedComponentStatistics:
    x: int
    """
    The leftmost (x) coordinate which is the inclusive start of the bounding box
    in the horizontal direction.
    """

    y: int
    """
    The topmost (y) coordinate which is the inclusive start of the bounding box
    in the vertical direction.
    """

    width: int
    """
    The horizontal size of the bounding box.
    """

    height: int
    """
    The vertical size of the bounding box.
    """

    area: int
    """
    The total area (in pixels) of the connected component.
    """

    center_x: float
    center_y: float


class ConnectedComponentsWithStatsResult(NamedTuple):
    number_of_labels: int
    labels: NDArray
    stats: NDArray
    centroids: NDArray

    def to_statistics(self) -> List[ConnectedComponentStatistics]:
        result = list()
        for stat, centroid in zip(self.stats.tolist(), self.centroids.tolist()):
            x, y, w, h, a = stat
            cx, cy = centroid
            result.append(ConnectedComponentStatistics(x, y, w, h, a, cx, cy))
        return result


def connected_components_with_stats(
    image: NDArray,
    connectivity=8,
    ltype: DataTypeLike = DataType.S32,
) -> ConnectedComponentsWithStatsResult:
    if connectivity in (4, 8):
        raise ValueError("The connectivity argument accepts only the values 8 or 4")

    _label_type = normalize_data_type(ltype)
    if _label_type not in (CV_32S, CV_16U):
        raise ValueError("The ltype argument accepts only the values CV_32S or CV_16U")

    result = cv2.connectedComponentsWithStats(
        image, None, None, None, connectivity, _label_type
    )

    retval, labels, stats, centroids = result
    return ConnectedComponentsWithStatsResult(retval, labels, stats, centroids)


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

    @staticmethod
    def cvl_connected_components(
        image: NDArray,
        connectivity=8,
        ltype: DataTypeLike = DataType.S32,
    ):
        return connected_components(image, connectivity, ltype)

    @staticmethod
    def cvl_connected_components_with_stats(
        image: NDArray,
        connectivity=8,
        ltype: DataTypeLike = DataType.S32,
    ):
        return connected_components_with_stats(image, connectivity, ltype)

# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum, unique
from typing import Iterable, List, NamedTuple, Sequence

import cv2
from numpy import int32, logical_and, uint8, zeros
from numpy.typing import NDArray

from cvlayer.cv.types.data_type import (
    CV_16U,
    CV_32S,
    DataType,
    DataTypeLike,
    normalize_data_type,
)
from cvlayer.cv.types.distance_transform_label import (
    DEFAULT_DISTANCE_TRANSFORM_LABEL,
    normalize_distance_transform_label,
)
from cvlayer.cv.types.distance_transform_mask import (
    DEFAULT_DISTANCE_TRANSFORM_MASK,
    normalize_distance_transform_mask,
)
from cvlayer.cv.types.distance_type import (
    DEFAULT_DISTANCE_TYPE,
    normalize_distance_type,
)
from cvlayer.typing import Image, PointF, RectI, SizeF


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
        raise ValueError("The ltype argument accepts only the values CV_32S or CV_16U")

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


def distance_transform(
    src: NDArray,
    distance_type=DEFAULT_DISTANCE_TYPE,
    mask_size=DEFAULT_DISTANCE_TRANSFORM_MASK,
    label_type=DEFAULT_DISTANCE_TRANSFORM_LABEL,
) -> NDArray:
    _distance_type = normalize_distance_type(distance_type)

    _mask_size = normalize_distance_transform_mask(mask_size)
    if _mask_size == cv2.DIST_MASK_PRECISE:
        raise ValueError("DIST_MASK_PRECISE is not supported by this variant")

    if _distance_type in (cv2.DIST_L1, cv2.DIST_C):
        # In case of the DIST_L1 or DIST_C distance type,
        # the parameter is forced to 3 because a 3×3 mask gives the same result as 5×5
        # or any larger aperture.
        _distance_type = cv2.DIST_MASK_3

    _label_type = normalize_distance_transform_label(label_type)
    return cv2.distanceTransform(src, _distance_type, _mask_size, None, _label_type)


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

    @staticmethod
    def cvl_distance_transform(
        src: NDArray,
        distance_type=DEFAULT_DISTANCE_TYPE,
        mask_size=DEFAULT_DISTANCE_TRANSFORM_MASK,
        label_type=DEFAULT_DISTANCE_TRANSFORM_LABEL,
    ):
        return distance_transform(src, distance_type, mask_size, label_type)

    @staticmethod
    def cvl_arc_length(curve: NDArray, closed=False):
        return arc_length(curve, closed)

    @staticmethod
    def cvl_approx_poly_dp(curve: NDArray, epsilon: float, closed=False):
        return approx_poly_dp(curve, epsilon, closed)

    @staticmethod
    def cvl_bounding_rect(array: NDArray):
        return bounding_rect(array)

    @staticmethod
    def cvl_min_area_rect(points: NDArray):
        return min_area_rect(points)

    @staticmethod
    def cvl_box_points(box: RotatedRect):
        return box_points(box)

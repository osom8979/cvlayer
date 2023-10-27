# -*- coding: utf-8 -*-

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.data_type import (
    CV_8U,
    CV_32F,
    DataType,
    DataTypeLike,
    normalize_data_type,
)
from cvlayer.cv.types.distance_transform_mask import (
    DEFAULT_DISTANCE_TRANSFORM_MASK,
    DIST_MASK_3,
    DIST_MASK_PRECISE,
    normalize_distance_transform_mask,
)
from cvlayer.cv.types.distance_type import (
    DEFAULT_DISTANCE_TYPE,
    DIST_C,
    DIST_L1,
    normalize_distance_type,
)


def distance_transform(
    src: NDArray,
    distance_type=DEFAULT_DISTANCE_TYPE,
    mask_size=DEFAULT_DISTANCE_TRANSFORM_MASK,
    dst_type: DataTypeLike = DataType.F32,
) -> NDArray:
    _distance_type = normalize_distance_type(distance_type)

    _mask_size = normalize_distance_transform_mask(mask_size)
    if _mask_size == DIST_MASK_PRECISE:
        raise ValueError("DIST_MASK_PRECISE is not supported by this variant")

    if _distance_type in (DIST_L1, DIST_C):
        # In case of the DIST_L1 or DIST_C distance type,
        # the parameter is forced to 3 because a 3x3 mask
        # gives the same result as 5x5 or any larger aperture.
        _mask_size = DIST_MASK_3

    _dst_type = normalize_data_type(dst_type)
    if _dst_type not in (CV_8U, CV_32F):
        raise ValueError("The dst_type argument accepts only CV_8U or CV_32F values")

    if _dst_type == CV_8U and _distance_type != DIST_L1:
        raise ValueError("Type CV_8U can be used only distance_type == DIST_L1")

    return cv2.distanceTransform(src, _distance_type, _mask_size, None, _dst_type)


class CvlTransformDistanceTransform:
    @staticmethod
    def cvl_distance_transform(
        src: NDArray,
        distance_type=DEFAULT_DISTANCE_TYPE,
        mask_size=DEFAULT_DISTANCE_TRANSFORM_MASK,
        dst_type: DataTypeLike = DataType.F32,
    ):
        return distance_transform(src, distance_type, mask_size, dst_type)

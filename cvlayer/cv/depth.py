# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Sequence, Union

from numpy import float32, float64, int16, ndarray, uint8, uint16
from numpy.typing import DTypeLike, NDArray

from cvlayer.cv.data_type import CV_16S, CV_32F, CV_64F

SAME_DEPTH_AS_SOURCE: Final[int] = -1

DEPTH_COMBINATIONS_TABLE: Final[Dict[DTypeLike, Sequence[int]]] = {
    uint8: (SAME_DEPTH_AS_SOURCE, CV_16S, CV_32F, CV_64F),
    uint16: (SAME_DEPTH_AS_SOURCE, CV_32F, CV_64F),
    int16: (SAME_DEPTH_AS_SOURCE, CV_32F, CV_64F),
    float32: (SAME_DEPTH_AS_SOURCE, CV_32F),
    float64: (SAME_DEPTH_AS_SOURCE, CV_64F),
}


@unique
class DesiredDepth(Enum):
    """desired depth of the destination image"""

    SAME_INPUT = SAME_DEPTH_AS_SOURCE
    INT16 = CV_16S
    FLOAT32 = CV_32F
    FLOAT64 = CV_64F


DEFAULT_DESIRED_DEPTH: Final[Union[DesiredDepth, int]] = DesiredDepth.SAME_INPUT


def normalize_desired_depth(depth: Optional[Union[DesiredDepth, int]] = None) -> int:
    if depth is None:
        return SAME_DEPTH_AS_SOURCE
    elif isinstance(depth, DesiredDepth):
        return depth.value
    elif isinstance(depth, int):
        return depth
    else:
        raise TypeError(f"Unsupported depth type: {type(depth).__name__}")


def validate_depth_combinations(
    src: Union[NDArray, DTypeLike],
    ddepth: Union[DesiredDepth, int],
) -> None:
    """
    https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#filter_depths
    """

    _type = src.dtype.type if isinstance(src, ndarray) else type(src)
    if _type not in DEPTH_COMBINATIONS_TABLE:
        raise TypeError(f"Unsupported input data type: {_type}")

    if normalize_desired_depth(ddepth) not in DEPTH_COMBINATIONS_TABLE[_type]:
        raise TypeError(f"Mismatch depth combination: {_type} -> {ddepth}")

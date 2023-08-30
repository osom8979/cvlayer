# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final

from numpy import float32, float64, int16, uint8, uint16

from cvlayer.cv.data_type import CV_16S, CV_32F, CV_64F

SAME_DEPTH_AS_SOURCE: Final[int] = -1

DEPTH_COMBINATIONS_TABLE = {
    uint8: (SAME_DEPTH_AS_SOURCE, CV_16S, CV_32F, CV_64F),
    uint16: (SAME_DEPTH_AS_SOURCE, CV_32F, CV_64F),
    int16: (SAME_DEPTH_AS_SOURCE, CV_32F, CV_64F),
    float32: (SAME_DEPTH_AS_SOURCE, CV_32F),
    float64: (SAME_DEPTH_AS_SOURCE, CV_64F),
}


def validate_depth_combinations(input_dtype, output_depth: int) -> None:
    """
    https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#filter_depths
    """

    if input_dtype not in DEPTH_COMBINATIONS_TABLE:
        raise TypeError(f"Unsupported input data type: {input_dtype}")

    if output_depth not in DEPTH_COMBINATIONS_TABLE[input_dtype]:
        raise TypeError(f"Mismatch depth combination: {input_dtype} -> {output_depth}")


@unique
class OutputDepth(Enum):
    SAME_INPUT = SAME_DEPTH_AS_SOURCE
    INT16 = CV_16S
    FLOAT32 = CV_32F
    FLOAT64 = CV_64F


DEFAULT_OUTPUT_DEPTH: Final[OutputDepth] = OutputDepth.SAME_INPUT

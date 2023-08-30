# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final

import cv2
from numpy import float32, float64, int16, uint8, uint16

CV_8U: Final[int] = cv2.CV_8U  # type: ignore[attr-defined]
CV_8S: Final[int] = cv2.CV_8S  # type: ignore[attr-defined]
assert CV_8U == 0
assert CV_8S == 1

CV_16U: Final[int] = cv2.CV_16U  # type: ignore[attr-defined]
CV_16S: Final[int] = cv2.CV_16S  # type: ignore[attr-defined]
assert CV_16U == 2
assert CV_16S == 3

assert not hasattr(cv2, "CV_32U")
CV_32S: Final[int] = cv2.CV_32S  # type: ignore[attr-defined]
assert CV_32S == 4

assert not hasattr(cv2, "CV_64U")
assert not hasattr(cv2, "CV_64S")

CV_32F: Final[int] = cv2.CV_32F  # type: ignore[attr-defined]
CV_64F: Final[int] = cv2.CV_64F  # type: ignore[attr-defined]
assert CV_32F == 5
assert CV_64F == 6

CV_16F: Final[int] = cv2.CV_16F  # type: ignore[attr-defined]
assert CV_16F == 7

SAME_DEPTH_AS_SOURCE: Final[int] = -1


@unique
class CvDataType(Enum):
    U8 = CV_8U
    S8 = CV_8S
    U16 = CV_16U
    S16 = CV_16S
    S32 = CV_32S
    F32 = CV_32F
    F64 = CV_64F
    F16 = CV_16F


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


if __name__ == "__main__":
    print(f"CV_8U = {CV_8U}")
    print(f"CV_8S = {CV_8S}")
    print(f"CV_16U = {CV_16U}")
    print(f"CV_16S = {CV_16S}")
    print(f"CV_32S = {CV_32S}")
    print(f"CV_32F = {CV_32F}")
    print(f"CV_64F = {CV_64F}")
    print(f"CV_16F = {CV_16F}")

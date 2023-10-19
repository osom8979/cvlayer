# -*- coding: utf-8 -*-

from typing import Any, Final, List, Optional, Sequence, Tuple, Type, overload

import cv2
from numpy import abs as np_abs
from numpy import float32, zeros_like
from numpy.typing import NDArray

from cvlayer.cv.norm import NormType
from cvlayer.typing import NumberT

SAME_DTYPE_AS_SRC: Final[int] = -1


# fmt: off
@overload
def _cast_numbers(nums: Sequence[Any], cls: Type[int]) -> List[int]: ...
@overload
def _cast_numbers(nums: Sequence[Any], cls: Type[float]) -> List[float]: ...
# fmt: on


def _cast_numbers(nums: Sequence[Any], cls: Type[NumberT]):
    if cls == float:
        return [float(n) for n in nums]
    elif cls == int:
        return [int(n) for n in nums]
    else:
        raise TypeError(f"Unsupported cls: {cls.__name__}")


def channels_min(array: NDArray, cls=int):
    shape = array.shape
    if len(shape) == 2:
        nums = [array.min()]
    elif len(shape) == 3:
        nums = [array[:, :, c].min() for c in range(shape[2])]
    else:
        raise ValueError(f"Unsupported array shape: {shape}")
    return _cast_numbers(nums, cls)


def channels_max(array: NDArray, cls=int):
    shape = array.shape
    if len(shape) == 2:
        nums = [array.max()]
    elif len(shape) == 3:
        nums = [array[:, :, c].max() for c in range(shape[2])]
    else:
        raise ValueError(f"Unsupported array shape: {shape}")
    return _cast_numbers(nums, cls)


def channels_mean(array: NDArray, cls=float):
    shape = array.shape
    if len(shape) == 2:
        nums = [array.mean()]
    elif len(shape) == 3:
        nums = [array[:, :, c].mean() for c in range(shape[2])]
    else:
        raise ValueError(f"Unsupported array shape: {shape}")
    return _cast_numbers(nums, cls)


def mean(
    src: NDArray,
    mask: Optional[NDArray] = None,
) -> Tuple[float, float, float, float]:
    means = cv2.mean(src, mask)
    assert len(means) == 4
    return means[0], means[1], means[2], means[3]


def channel_mean_abs_diff(src: NDArray) -> NDArray[float32]:
    b, g, r = cv2.split(float32(src))
    bg = np_abs(b - g)
    gr = np_abs(g - r)
    rb = np_abs(r - b)
    return (bg + gr + rb) / 3.0


def split(m: NDArray) -> Sequence[NDArray]:
    return cv2.split(m)


def merge(mv: Sequence[NDArray]) -> NDArray:
    return cv2.merge(mv)


def add(*arrays: NDArray) -> NDArray:
    result = zeros_like(arrays[0])
    for a in arrays:
        result = cv2.add(result, a)
    return result


def normalize(
    src: NDArray,
    alpha=1.0,
    beta=0.0,
    norm_type=NormType.L2,
    dtype=SAME_DTYPE_AS_SRC,
    mask: Optional[NDArray] = None,
):
    dst = zeros_like(src)
    cv2.normalize(
        src,
        dst,
        alpha=alpha,
        beta=beta,
        norm_type=norm_type.value,
        dtype=dtype,
        mask=mask,
    )
    return dst


def normalize_uint8_minmax(
    src: NDArray,
    dtype=SAME_DTYPE_AS_SRC,
    mask: Optional[NDArray] = None,
):
    return normalize(src, 0, 255, NormType.MINMAX, dtype, mask)


# def abs():
# def absdiff():
# def calcCovarMatrix():
# def cartToPolar():
# def checkRange():
# def compare():
# def completeSymm():
# def convertScaleAbs():
# def countNonZero():
# def cvarrToMat():
# def dct():
# def dft():
# def divide():
# def determinant():
# def eigen():
# def exp():
# def extractImageCOI():
# def insertImageCOI():
# def flip():
# def gemm():
# def getConvertElem():
# def getOptimalDFTSize():
# def idct():
# def idft():
# def inRange():
# def invert():
# def log():
# def LUT():
# def magnitude():
# def Mahalanobis():
# def meanStdDev():
# def minMaxIdx():
# def minMaxLoc():
# def mixChannels():
# def mulSpectrums():
# def multiply():
# def mulTransposed():
# def norm():
# def perspectiveTransform():
# def phase():
# def polarToCart():
# def pow():
# def randu():
# def randn():
# def randShuffle():
# def reduce():
# def repeat():
# def scaleAdd():
# def setIdentity():
# def solve():
# def solveCubic():
# def solvePoly():
# def sort():
# def sortIdx():
# def sqrt():
# def subtract():
# def sum():
# def theRNG():
# def trace():
# def transform():
# def transpose():


class CvlBasic:
    @staticmethod
    def cvl_channels_min(array: NDArray, cls=int):
        return channels_min(array, cls)

    @staticmethod
    def cvl_channels_max(array: NDArray, cls=int):
        return channels_max(array, cls)

    @staticmethod
    def cvl_channels_mean(array: NDArray, cls=float):
        return channels_mean(array, cls)

    @staticmethod
    def cvl_mean(src: NDArray, mask: Optional[NDArray] = None):
        return mean(src, mask)

    @staticmethod
    def cvl_channel_mean_abs_diff(src: NDArray):
        return channel_mean_abs_diff(src)

    @staticmethod
    def cvl_split(m: NDArray):
        return split(m)

    @staticmethod
    def cvl_merge(mv: Sequence[NDArray]):
        return merge(mv)

    @staticmethod
    def cvl_add(*arrays: NDArray):
        return add(*arrays)

    @staticmethod
    def cvl_normalize(
        src: NDArray,
        alpha=1.0,
        beta=0.0,
        norm_type=NormType.L2,
        dtype=-1,
        mask: Optional[NDArray] = None,
    ):
        return normalize(src, alpha, beta, norm_type, dtype, mask)

    @staticmethod
    def cvl_normalize_uint8_minmax(
        src: NDArray,
        dtype=SAME_DTYPE_AS_SRC,
        mask: Optional[NDArray] = None,
    ):
        return normalize_uint8_minmax(src, dtype, mask)

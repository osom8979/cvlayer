# -*- coding: utf-8 -*-

from typing import Optional

import cv2
from numpy.typing import NDArray


def bitwise_and(
    src1: NDArray,
    src2: NDArray,
    mask: Optional[NDArray] = None,
) -> NDArray:
    return cv2.bitwise_and(src1, src2, mask=mask)


def bitwise_or(
    src1: NDArray,
    src2: NDArray,
    mask: Optional[NDArray] = None,
) -> NDArray:
    return cv2.bitwise_or(src1, src2, mask=mask)


def bitwise_xor(
    src1: NDArray,
    src2: NDArray,
    mask: Optional[NDArray] = None,
) -> NDArray:
    return cv2.bitwise_xor(src1, src2, mask=mask)


def bitwise_not(
    src: NDArray,
    mask: Optional[NDArray] = None,
) -> NDArray:
    return cv2.bitwise_not(src, mask=mask)


class CvlBitwise:
    @staticmethod
    def cvl_bitwise_and(src1: NDArray, src2: NDArray, mask: Optional[NDArray] = None):
        return bitwise_and(src1, src2, mask)

    @staticmethod
    def cvl_bitwise_or(src1: NDArray, src2: NDArray, mask: Optional[NDArray] = None):
        return bitwise_or(src1, src2, mask)

    @staticmethod
    def cvl_bitwise_xor(src1: NDArray, src2: NDArray, mask: Optional[NDArray] = None):
        return bitwise_xor(src1, src2, mask)

    @staticmethod
    def cvl_bitwise_not(src: NDArray, mask: Optional[NDArray] = None):
        return bitwise_not(src, mask)

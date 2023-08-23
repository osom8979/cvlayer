# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.bitwise import bitwise_and, bitwise_not, bitwise_or, bitwise_xor


class CvlBitwise:
    @staticmethod
    def cvl_bitwise_and(src1: NDArray, src2: NDArray, mask: NDArray) -> NDArray:
        return bitwise_and(src1, src2, mask)

    @staticmethod
    def cvl_bitwise_or(src1: NDArray, src2: NDArray, mask: NDArray) -> NDArray:
        return bitwise_or(src1, src2, mask)

    @staticmethod
    def cvl_bitwise_xor(src1: NDArray, src2: NDArray, mask: NDArray) -> NDArray:
        return bitwise_xor(src1, src2, mask)

    @staticmethod
    def cvl_bitwise_not(src: NDArray, mask: NDArray) -> NDArray:
        return bitwise_not(src, mask)

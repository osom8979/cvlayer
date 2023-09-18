# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import cv2
from numpy.typing import NDArray


def mean(
    src: NDArray,
    mask: Optional[NDArray] = None,
) -> Tuple[float, float, float, float]:
    means = cv2.mean(src, mask)
    assert len(means) == 4
    return means[0], means[1], means[2], means[3]


# def abs():
# def absdiff():
# def add():
# def addWeighted():
# def bitwise_and():
# def bitwise_not():
# def bitwise_or():
# def bitwise_xor():
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
# def max():
# def mean():
# def meanStdDev():
# def merge():
# def min():
# def minMaxIdx():
# def minMaxLoc():
# def mixChannels():
# def mulSpectrums():
# def multiply():
# def mulTransposed():
# def norm():
# def normalize():
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
# def split():
# def sqrt():
# def subtract():
# def sum():
# def theRNG():
# def trace():
# def transform():
# def transpose():


class CvlBasic:
    @staticmethod
    def cvl_mean(src: NDArray, mask: Optional[NDArray] = None):
        return mean(src, mask)

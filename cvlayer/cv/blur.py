# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

from cvlayer.cv.border import DEFAULT_BORDER_TYPE
from cvlayer.typing import PointI, SizeI

DEFAULT_KSIZE: Final[SizeI] = (3, 3)
DEFAULT_ANCHOR: Final[PointI] = (-1, -1)

DEFAULT_GAUSSIAN_BLUR_SIGMA_X: Final[float] = 0.0
"""
Gaussian kernel standard deviation in X direction.
"""

DEFAULT_GAUSSIAN_BLUR_SIGMA_Y: Final[float] = 0.0
"""
Gaussian kernel standard deviation in Y direction;
if sigmaY is zero, it is set to be equal to sigmaX,
if both sigmas are zeros, they are computed from ksize.width and ksize.height,
respectively (see getGaussianKernel for details);
to fully control the result regardless of possible future modifications of all this
semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
"""

DEFAULT_BILATERAL_FILTER_D: Final[int] = 9
"""
Diameter of each pixel neighborhood that is used during filtering.
If it is non-positive, it is computed from sigmaSpace.
"""

DEFAULT_BILATERAL_FILTER_SIGMA_COLOR: Final[float] = 75.0
"""
Filter sigma in the color space.
A larger value of the parameter means that farther colors within the pixel neighborhood
(see sigmaSpace) will be mixed together,
resulting in larger areas of semi-equal color.
"""

DEFAULT_BILATERAL_FILTER_SIGMA_SPACE: Final[float] = 75.0
"""
Filter sigma in the coordinate space.
A larger value of the parameter means that farther pixels will influence each other as
long as their colors are close enough (see sigmaColor ).
When d>0, it specifies the neighborhood size regardless of sigmaSpace.
Otherwise, d is proportional to sigmaSpace.
"""


def blur(
    src: NDArray,
    ksize=DEFAULT_KSIZE,
    anchor=DEFAULT_ANCHOR,
    border_type=DEFAULT_BORDER_TYPE,
) -> NDArray:
    return cv2.blur(src, ksize, anchor=anchor, borderType=border_type.value)


def median_blur(src: NDArray, ksize=DEFAULT_KSIZE) -> NDArray:
    return cv2.medianBlur(src, ksize)


def gaussian_blur(
    src: NDArray,
    ksize=DEFAULT_KSIZE,
    sigma_x=DEFAULT_GAUSSIAN_BLUR_SIGMA_X,
    sigma_y=DEFAULT_GAUSSIAN_BLUR_SIGMA_Y,
    border_type=DEFAULT_BORDER_TYPE,
) -> NDArray:
    return cv2.GaussianBlur(
        src,
        ksize,
        sigmaX=sigma_x,
        dst=None,
        sigmaY=sigma_y,
        borderType=border_type.value,
    )


def bilateral_filter(
    src: NDArray,
    d=DEFAULT_BILATERAL_FILTER_D,
    sigma_color=DEFAULT_BILATERAL_FILTER_SIGMA_COLOR,
    sigma_space=DEFAULT_BILATERAL_FILTER_SIGMA_SPACE,
    border_type=DEFAULT_BORDER_TYPE,
) -> NDArray:
    return cv2.bilateralFilter(
        src,
        d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
        dst=None,
        borderType=border_type.value,
    )


class CvlBlur:
    @staticmethod
    def cvl_blur(
        src: NDArray,
        ksize=DEFAULT_KSIZE,
        anchor=DEFAULT_ANCHOR,
        border_type=DEFAULT_BORDER_TYPE,
    ):
        return blur(src, ksize, anchor, border_type)

    @staticmethod
    def cvl_median_blur(src: NDArray, ksize=DEFAULT_KSIZE):
        return median_blur(src, ksize)

    @staticmethod
    def cvl_gaussian_blur(
        src: NDArray,
        ksize=DEFAULT_KSIZE,
        sigma_x=DEFAULT_GAUSSIAN_BLUR_SIGMA_X,
        sigma_y=DEFAULT_GAUSSIAN_BLUR_SIGMA_Y,
        border_type=DEFAULT_BORDER_TYPE,
    ):
        return gaussian_blur(src, ksize, sigma_x, sigma_y, border_type)

    @staticmethod
    def cvl_bilateral_filter(
        src: NDArray,
        d=DEFAULT_BILATERAL_FILTER_D,
        sigma_color=DEFAULT_BILATERAL_FILTER_SIGMA_COLOR,
        sigma_space=DEFAULT_BILATERAL_FILTER_SIGMA_SPACE,
        border_type=DEFAULT_BORDER_TYPE,
    ):
        return bilateral_filter(src, d, sigma_color, sigma_space, border_type)

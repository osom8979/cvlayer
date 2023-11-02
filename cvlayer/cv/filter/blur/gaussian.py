# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.border import DEFAULT_BORDER_TYPE, normalize_border_type
from cvlayer.typing import SizeI

DEFAULT_KSIZE: Final[SizeI] = (3, 3)

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


def gaussian_blur(
    src: NDArray,
    ksize=DEFAULT_KSIZE,
    sigma_x=DEFAULT_GAUSSIAN_BLUR_SIGMA_X,
    sigma_y=DEFAULT_GAUSSIAN_BLUR_SIGMA_Y,
    border=DEFAULT_BORDER_TYPE,
) -> NDArray:
    _border = normalize_border_type(border)
    return cv2.GaussianBlur(
        src,
        ksize,
        sigmaX=sigma_x,
        dst=None,
        sigmaY=sigma_y,
        borderType=_border,
    )


class CvlFilterBlurGaussian:
    @staticmethod
    def cvl_gaussian_blur(
        src: NDArray,
        ksize=DEFAULT_KSIZE,
        sigma_x=DEFAULT_GAUSSIAN_BLUR_SIGMA_X,
        sigma_y=DEFAULT_GAUSSIAN_BLUR_SIGMA_Y,
        border=DEFAULT_BORDER_TYPE,
    ):
        return gaussian_blur(src, ksize, sigma_x, sigma_y, border)

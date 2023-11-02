# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.border import DEFAULT_BORDER_TYPE, normalize_border_type

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


def bilateral_filter(
    src: NDArray,
    d=DEFAULT_BILATERAL_FILTER_D,
    sigma_color=DEFAULT_BILATERAL_FILTER_SIGMA_COLOR,
    sigma_space=DEFAULT_BILATERAL_FILTER_SIGMA_SPACE,
    border=DEFAULT_BORDER_TYPE,
) -> NDArray:
    _border = normalize_border_type(border)
    return cv2.bilateralFilter(
        src,
        d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
        dst=None,
        borderType=_border,
    )


class CvlFilterBlurBilateral:
    @staticmethod
    def cvl_bilateral_filter(
        src: NDArray,
        d=DEFAULT_BILATERAL_FILTER_D,
        sigma_color=DEFAULT_BILATERAL_FILTER_SIGMA_COLOR,
        sigma_space=DEFAULT_BILATERAL_FILTER_SIGMA_SPACE,
        border=DEFAULT_BORDER_TYPE,
    ):
        return bilateral_filter(src, d, sigma_color, sigma_space, border)

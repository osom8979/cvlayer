# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

DEFAULT_H: Final[float] = 10.0
"""
parameter deciding filter strength.
Higher h value removes noise better, but removes details of image also.
(10 is ok)
"""

DEFAULT_H_FOR_COLOR_COMPONENTS: Final[float] = DEFAULT_H
"""same as h, but for color images only. (normally same as h)"""

DEFAULT_TEMPLATE_WINDOW_SIZE: Final[int] = 7
"""Should be odd. (recommended 7)"""

DEFAULT_SEARCH_WINDOW_SIZE: Final[int] = 21
"""Should be odd. (recommended 21)"""


def fast_nl_means_denoising(
    src: NDArray,
    h=DEFAULT_H,
    template_window_size=DEFAULT_TEMPLATE_WINDOW_SIZE,
    search_window_size=DEFAULT_SEARCH_WINDOW_SIZE,
) -> NDArray:
    """
    Works with a single grayscale images.
    """
    return cv2.fastNlMeansDenoising(
        src,
        None,
        h,
        template_window_size,
        search_window_size,
    )


def fast_nl_means_denoising_colored(
    src: NDArray,
    h=DEFAULT_H,
    h_color=DEFAULT_H_FOR_COLOR_COMPONENTS,
    template_window_size=DEFAULT_TEMPLATE_WINDOW_SIZE,
    search_window_size=DEFAULT_SEARCH_WINDOW_SIZE,
) -> NDArray:
    """
    works with a color image.
    """
    return cv2.fastNlMeansDenoisingColored(
        src,
        None,
        h,
        h_color,
        template_window_size,
        search_window_size,
    )


# cv2.fastNlMeansDenoisingMulti()
# cv2.fastNlMeansDenoisingColoredMulti()


class CvlNimd:
    """
    Non-Local Means Denoising algorithms.
    """

    @staticmethod
    def cvl_fast_nl_means_denoising(
        src: NDArray,
        h=DEFAULT_H,
        template_window_size=DEFAULT_TEMPLATE_WINDOW_SIZE,
        search_window_size=DEFAULT_SEARCH_WINDOW_SIZE,
    ):
        return fast_nl_means_denoising(src, h, template_window_size, search_window_size)

    @staticmethod
    def cvl_fast_nl_means_denoising_colored(
        src: NDArray,
        h=DEFAULT_H,
        h_color=DEFAULT_H_FOR_COLOR_COMPONENTS,
        template_window_size=DEFAULT_TEMPLATE_WINDOW_SIZE,
        search_window_size=DEFAULT_SEARCH_WINDOW_SIZE,
    ):
        return fast_nl_means_denoising_colored(
            src, h, h_color, template_window_size, search_window_size
        )

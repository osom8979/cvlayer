# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.border import DEFAULT_BORDER_TYPE, normalize_border_type
from cvlayer.typing import PointI, SizeI

DEFAULT_KSIZE: Final[SizeI] = (3, 3)
DEFAULT_ANCHOR: Final[PointI] = (-1, -1)


def blur(
    src: NDArray,
    ksize=DEFAULT_KSIZE,
    anchor=DEFAULT_ANCHOR,
    border=DEFAULT_BORDER_TYPE,
) -> NDArray:
    _border = normalize_border_type(border)
    return cv2.blur(src, ksize, anchor=anchor, borderType=_border)


class CvlFilterBlurBlur:
    @staticmethod
    def cvl_blur(
        src: NDArray,
        ksize=DEFAULT_KSIZE,
        anchor=DEFAULT_ANCHOR,
        border=DEFAULT_BORDER_TYPE,
    ):
        return blur(src, ksize, anchor, border)

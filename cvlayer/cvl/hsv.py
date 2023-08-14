# -*- coding: utf-8 -*-

from colorsys import hsv_to_rgb, rgb_to_hsv
from typing import Tuple

from cvlayer.cv.hsv import (
    hsv_hue_as_cv_unit,
    hsv_saturation_as_cv_unit,
    hsv_value_as_cv_unit,
)


class CvlHsv:
    @staticmethod
    def cvl_hsv_hue_as_cv_unit(h: float) -> int:
        return hsv_hue_as_cv_unit(h)

    @staticmethod
    def cvl_hsv_saturation_as_cv_unit(s: float) -> int:
        return hsv_saturation_as_cv_unit(s)

    @staticmethod
    def cvl_hsv_value_as_cv_unit(v: float) -> int:
        return hsv_value_as_cv_unit(v)

    @staticmethod
    def cvl_hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
        return hsv_to_rgb(h, s, v)

    @staticmethod
    def cvl_rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
        return rgb_to_hsv(r, g, b)

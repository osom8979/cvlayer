# -*- coding: utf-8 -*-

from colorsys import hsv_to_rgb, rgb_to_hsv
from math import floor
from typing import Final

HSV_HUE_MIN: Final[int] = 0
HSV_HUE_MAX: Final[int] = 179

HSV_SATURATION_MIN: Final[int] = 0
HSV_SATURATION_MAX: Final[int] = 255

HSV_VALUE_MIN: Final[int] = 0
HSV_VALUE_MAX: Final[int] = 255


def hsv_hue_as_cv_unit(h: float) -> int:
    return floor(h / 360.0 * HSV_HUE_MAX)


def hsv_saturation_as_cv_unit(s: float) -> int:
    return floor(s / 100.0 * HSV_SATURATION_MAX)


def hsv_value_as_cv_unit(v: float) -> int:
    return floor(v / 100.0 * HSV_VALUE_MAX)


class CvlHsv:
    @staticmethod
    def cvl_hsv_hue_as_cv_unit(h: float):
        return hsv_hue_as_cv_unit(h)

    @staticmethod
    def cvl_hsv_saturation_as_cv_unit(s: float):
        return hsv_saturation_as_cv_unit(s)

    @staticmethod
    def cvl_hsv_value_as_cv_unit(v: float):
        return hsv_value_as_cv_unit(v)

    @staticmethod
    def cvl_hsv_to_rgb(h: float, s: float, v: float):
        return hsv_to_rgb(h, s, v)

    @staticmethod
    def cvl_rgb_to_hsv(r: float, g: float, b: float):
        return rgb_to_hsv(r, g, b)

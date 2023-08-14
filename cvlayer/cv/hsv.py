# -*- coding: utf-8 -*-

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

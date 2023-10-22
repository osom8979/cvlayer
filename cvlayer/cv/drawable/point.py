# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.drawable.circle import draw_circle_coord
from cvlayer.cv.drawable.defaults import (
    DEFAULT_COLOR,
    DEFAULT_LINE_TYPE,
    DEFAULT_RADIUS,
    DEFAULT_SHIFT,
)
from cvlayer.cv.types.thickness import FILLED
from cvlayer.typing import Number, PointN


def draw_point_coord(
    image: NDArray,
    x: Number,
    y: Number,
    radius=DEFAULT_RADIUS,
    color=DEFAULT_COLOR,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    return draw_circle_coord(image, x, y, radius, color, FILLED, line, shift)


def draw_point(
    image: NDArray,
    center: PointN,
    radius=DEFAULT_RADIUS,
    color=DEFAULT_COLOR,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    return draw_point_coord(image, center[0], center[1], radius, color, line, shift)


class CvlDrawablePoint:
    @staticmethod
    def cvl_draw_point_coord(
        image: NDArray,
        x: Number,
        y: Number,
        radius=DEFAULT_RADIUS,
        color=DEFAULT_COLOR,
        line=DEFAULT_LINE_TYPE,
        shift=DEFAULT_SHIFT,
    ):
        return draw_point_coord(image, x, y, radius, color, line, shift)

    @staticmethod
    def cvl_draw_point(
        image: NDArray,
        center: PointN,
        radius=DEFAULT_RADIUS,
        color=DEFAULT_COLOR,
        line=DEFAULT_LINE_TYPE,
        shift=DEFAULT_SHIFT,
    ):
        return draw_point(image, center, radius, color, line, shift)

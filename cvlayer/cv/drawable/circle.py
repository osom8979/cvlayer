# -*- coding: utf-8 -*-

import cv2
from numpy.typing import NDArray

from cvlayer.cv.drawable.defaults import (
    DEFAULT_COLOR,
    DEFAULT_LINE_TYPE,
    DEFAULT_RADIUS,
    DEFAULT_SHIFT,
    DEFAULT_THICKNESS,
)
from cvlayer.cv.types.color import normalize_color
from cvlayer.cv.types.line_type import normalize_line_type
from cvlayer.typing import Number, PointN


def draw_circle_coord(
    image: NDArray,
    x: Number,
    y: Number,
    radius=DEFAULT_RADIUS,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    _center = int(x), int(y)
    _color = normalize_color(color)
    _line = normalize_line_type(line)
    return cv2.circle(image, _center, radius, _color, thickness, _line, shift)


def draw_circle(
    image: NDArray,
    center: PointN,
    radius=DEFAULT_RADIUS,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    return draw_circle_coord(
        image,
        center[0],
        center[1],
        radius,
        color,
        thickness,
        line,
        shift,
    )


class CvlDrawableCircle:
    @staticmethod
    def cvl_draw_circle_coord(
        image: NDArray,
        x: Number,
        y: Number,
        radius=DEFAULT_RADIUS,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        shift=DEFAULT_SHIFT,
    ):
        return draw_circle_coord(image, x, y, radius, color, thickness, line, shift)

    @staticmethod
    def cvl_draw_circle(
        image: NDArray,
        center: PointN,
        radius=DEFAULT_RADIUS,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        shift=DEFAULT_SHIFT,
    ):
        return draw_circle(image, center, radius, color, thickness, line, shift)

# -*- coding: utf-8 -*-

import cv2
from numpy.typing import NDArray

from cvlayer.cv.drawable.defaults import (
    DEFAULT_COLOR,
    DEFAULT_LINE_TYPE,
    DEFAULT_SHIFT,
    DEFAULT_THICKNESS,
)
from cvlayer.cv.types.color import normalize_color
from cvlayer.cv.types.line_type import normalize_line_type
from cvlayer.typing import Number, PointN


def draw_line_coord(
    image: NDArray,
    x1: Number,
    y1: Number,
    x2: Number,
    y2: Number,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    p1 = int(x1), int(y1)
    p2 = int(x2), int(y2)
    _color = normalize_color(color)
    _line = normalize_line_type(line)
    return cv2.line(image, p1, p2, _color, thickness, _line, shift)


def draw_line(
    image: NDArray,
    point1: PointN,
    point2: PointN,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    x1, y1 = point1
    x2, y2 = point2
    return draw_line_coord(image, x1, y1, x2, y2, color, thickness, line, shift)


class CvlDrawableLine:
    @staticmethod
    def cvl_draw_line_coord(
        image: NDArray,
        x1: Number,
        y1: Number,
        x2: Number,
        y2: Number,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        shift=DEFAULT_SHIFT,
    ):
        return draw_line_coord(image, x1, y1, x2, y2, color, thickness, line, shift)

    @staticmethod
    def cvl_draw_line(
        image: NDArray,
        point1: PointN,
        point2: PointN,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        shift=DEFAULT_SHIFT,
    ):
        return draw_line(image, point1, point2, color, thickness, line, shift)

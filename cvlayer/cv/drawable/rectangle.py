# -*- coding: utf-8 -*-

import cv2
from numpy.typing import NDArray

from cvlayer.cv.color import normalize_color
from cvlayer.cv.drawable.defaults import (
    DEFAULT_COLOR,
    DEFAULT_LINE_TYPE,
    DEFAULT_SHIFT,
    DEFAULT_THICKNESS,
)
from cvlayer.cv.types.line_type import normalize_line
from cvlayer.typing import Number, RectN


def draw_rectangle_coord(
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
    _line = normalize_line(line)
    return cv2.rectangle(image, p1, p2, _color, thickness, _line, shift)


def draw_rectangle(
    image: NDArray,
    roi: RectN,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    x1, y1, x2, y2 = roi
    return draw_rectangle_coord(image, x1, y1, x2, y2, color, thickness, line, shift)


class CvlDrawableRectangle:
    @staticmethod
    def cvl_draw_rectangle_coord(
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
        return draw_rectangle_coord(
            image, x1, y1, x2, y2, color, thickness, line, shift
        )

    @staticmethod
    def cvl_draw_rectangle(
        image: NDArray,
        roi: RectN,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        shift=DEFAULT_SHIFT,
    ):
        return draw_rectangle(image, roi, color, thickness, line, shift)

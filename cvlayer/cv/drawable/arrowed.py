# -*- coding: utf-8 -*-

from typing import Final

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

DEFAULT_ARROWED_LINE_TIP_LENGTH: Final[float] = 0.1


def draw_arrowed_coord(
    image: NDArray,
    x1: Number,
    y1: Number,
    x2: Number,
    y2: Number,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
    tip=DEFAULT_ARROWED_LINE_TIP_LENGTH,
) -> NDArray:
    p1 = int(x1), int(y1)
    p2 = int(x2), int(y2)
    _color = normalize_color(color)
    _line = normalize_line_type(line)
    return cv2.arrowedLine(image, p1, p2, _color, thickness, _line, shift, tip)


def draw_arrowed(
    image: NDArray,
    p1: PointN,
    p2: PointN,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
    tip=DEFAULT_ARROWED_LINE_TIP_LENGTH,
) -> NDArray:
    x1, y1 = p1
    x2, y2 = p2
    return draw_arrowed_coord(image, x1, y1, x2, y2, color, thickness, line, shift, tip)


class CvlDrawableArrowed:
    @staticmethod
    def cvl_draw_arrowed_coord(
        image: NDArray,
        x1: Number,
        y1: Number,
        x2: Number,
        y2: Number,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        shift=DEFAULT_SHIFT,
        tip=DEFAULT_ARROWED_LINE_TIP_LENGTH,
    ):
        return draw_arrowed_coord(
            image, x1, y1, x2, y2, color, thickness, line, shift, tip
        )

    @staticmethod
    def cvl_draw_arrowed(
        image: NDArray,
        p1: PointN,
        p2: PointN,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        shift=DEFAULT_SHIFT,
        tip=DEFAULT_ARROWED_LINE_TIP_LENGTH,
    ):
        return draw_arrowed(image, p1, p2, color, thickness, line, shift, tip)

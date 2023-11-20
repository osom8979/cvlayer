# -*- coding: utf-8 -*-

from typing import Final, Union

import cv2
from numpy.typing import NDArray

from cvlayer.cv.drawable.defaults import (
    DEFAULT_COLOR,
    DEFAULT_LINE_TYPE,
    DEFAULT_THICKNESS,
)
from cvlayer.cv.types.color import normalize_color
from cvlayer.cv.types.line_type import normalize_line_type
from cvlayer.cv.types.marker import MarkerType, normalize_marker
from cvlayer.typing import Number, PointN

DEFAULT_MARKER_SIZE: Final[int] = 20
DEFAULT_MARKER_TYPE: Final[Union[MarkerType, int]] = MarkerType.CROSS


def draw_marker_coord(
    image: NDArray,
    x: Number,
    y: Number,
    size=DEFAULT_MARKER_SIZE,
    marker=DEFAULT_MARKER_TYPE,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
) -> NDArray:
    position = int(x), int(y)
    _color = normalize_color(color)
    _line = normalize_line_type(line)
    _marker = normalize_marker(marker)
    return cv2.drawMarker(
        image,
        position,
        _color,
        _marker,
        size,
        thickness,
        _line,
    )


def draw_marker(
    image: NDArray,
    pos: PointN,
    size=DEFAULT_MARKER_SIZE,
    marker=DEFAULT_MARKER_TYPE,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
) -> NDArray:
    x, y = pos
    return draw_marker_coord(image, x, y, color, marker, size, thickness, line)


class CvlDrawableMarker:
    @staticmethod
    def cvl_draw_marker_coord(
        image: NDArray,
        x: Number,
        y: Number,
        size=DEFAULT_MARKER_SIZE,
        marker=DEFAULT_MARKER_TYPE,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
    ):
        return draw_marker_coord(image, x, y, size, marker, color, thickness, line)

    @staticmethod
    def cvl_draw_marker(
        image: NDArray,
        pos: PointN,
        size=DEFAULT_MARKER_SIZE,
        marker=DEFAULT_MARKER_TYPE,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
    ):
        return draw_marker(image, pos, size, marker, color, thickness, line)

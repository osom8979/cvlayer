# -*- coding: utf-8 -*-

import cv2
from numpy.typing import NDArray

from cvlayer.cv.drawable.defaults import (
    DEFAULT_FONT_COLOR,
    DEFAULT_FONT_FACE,
    DEFAULT_FONT_SCALE,
    DEFAULT_LINE_TYPE,
    DEFAULT_TEXT_ORIGIN,
    DEFAULT_THICKNESS,
)
from cvlayer.cv.types.color import normalize_color
from cvlayer.cv.types.font_face import normalize_font_face
from cvlayer.cv.types.line_type import normalize_line_type
from cvlayer.cv.types.text_origin import normalize_text_origin
from cvlayer.typing import Number, PointN


def draw_text_coord(
    image: NDArray,
    text: str,
    x: Number,
    y: Number,
    font=DEFAULT_FONT_FACE,
    scale=DEFAULT_FONT_SCALE,
    color=DEFAULT_FONT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    origin=DEFAULT_TEXT_ORIGIN,
) -> NDArray:
    org = int(x), int(y)
    _font = normalize_font_face(font)
    _color = normalize_color(color)
    _line = normalize_line_type(line)
    _origin = normalize_text_origin(origin)
    return cv2.putText(
        image, text, org, _font, scale, _color, thickness, _line, _origin
    )


def draw_text(
    image: NDArray,
    text: str,
    pos: PointN,
    font=DEFAULT_FONT_FACE,
    scale=DEFAULT_FONT_SCALE,
    color=DEFAULT_FONT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    origin=DEFAULT_TEXT_ORIGIN,
) -> NDArray:
    x, y = pos
    return draw_text_coord(
        image, text, x, y, font, scale, color, thickness, line, origin
    )


class CvlDrawableTextText:
    @staticmethod
    def cvl_draw_text_coord(
        image: NDArray,
        text: str,
        x: Number,
        y: Number,
        font=DEFAULT_FONT_FACE,
        scale=DEFAULT_FONT_SCALE,
        color=DEFAULT_FONT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        origin=DEFAULT_TEXT_ORIGIN,
    ):
        return draw_text_coord(
            image,
            text,
            x,
            y,
            font,
            scale,
            color,
            thickness,
            line,
            origin,
        )

    @staticmethod
    def cvl_draw_text(
        image: NDArray,
        text: str,
        pos: PointN,
        font=DEFAULT_FONT_FACE,
        scale=DEFAULT_FONT_SCALE,
        color=DEFAULT_FONT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        origin=DEFAULT_TEXT_ORIGIN,
    ):
        return draw_text(image, text, pos, font, scale, color, thickness, line, origin)

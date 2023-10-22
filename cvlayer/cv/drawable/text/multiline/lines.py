# -*- coding: utf-8 -*-

from typing import List

from numpy.typing import NDArray

from cvlayer.cv.drawable.defaults import (
    DEFAULT_FONT_FACE,
    DEFAULT_FONT_SCALE,
    DEFAULT_LINE_TYPE,
    DEFAULT_TEXT_ORIGIN,
    DEFAULT_THICKNESS,
    MULTILINE_COLOR,
    MULTILINE_LINE_SPACING,
)
from cvlayer.cv.drawable.text.multiline.measure import LineTextSize
from cvlayer.cv.drawable.text.text import draw_text_coord
from cvlayer.typing import Number, PointN


def draw_multiline_text_lines_coord(
    image: NDArray,
    lines: List[LineTextSize],
    x: Number,
    y: Number,
    font=DEFAULT_FONT_FACE,
    scale=DEFAULT_FONT_SCALE,
    color=MULTILINE_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    spacing=MULTILINE_LINE_SPACING,
    origin=DEFAULT_TEXT_ORIGIN,
) -> NDArray:
    for lts in lines:
        text = lts.text
        width, height = lts.size
        baseline = lts.baseline
        y += height
        draw_text_coord(image, text, x, y, font, scale, color, thickness, line, origin)
        y += baseline + spacing
    return image


def draw_multiline_text_lines(
    image: NDArray,
    lines: List[LineTextSize],
    pos: PointN,
    font=DEFAULT_FONT_FACE,
    scale=DEFAULT_FONT_SCALE,
    color=MULTILINE_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    spacing=MULTILINE_LINE_SPACING,
    origin=DEFAULT_TEXT_ORIGIN,
) -> NDArray:
    return draw_multiline_text_lines_coord(
        image,
        lines,
        pos[0],
        pos[1],
        font,
        scale,
        color,
        thickness,
        line,
        spacing,
        origin,
    )


class CvlDrawableTextMultilineLines:
    @staticmethod
    def cvl_draw_multiline_text_lines_coord(
        image: NDArray,
        lines: List[LineTextSize],
        x: Number,
        y: Number,
        font=DEFAULT_FONT_FACE,
        scale=DEFAULT_FONT_SCALE,
        color=MULTILINE_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        spacing=MULTILINE_LINE_SPACING,
        origin=DEFAULT_TEXT_ORIGIN,
    ):
        return draw_multiline_text_lines_coord(
            image,
            lines,
            x,
            y,
            font,
            scale,
            color,
            thickness,
            line,
            spacing,
            origin,
        )

    @staticmethod
    def cvl_draw_multiline_text_lines(
        image: NDArray,
        lines: List[LineTextSize],
        pos: PointN,
        font=DEFAULT_FONT_FACE,
        scale=DEFAULT_FONT_SCALE,
        color=MULTILINE_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        spacing=MULTILINE_LINE_SPACING,
        origin=DEFAULT_TEXT_ORIGIN,
    ):
        return draw_multiline_text_lines(
            image,
            lines,
            pos,
            font,
            scale,
            color,
            thickness,
            line,
            spacing,
            origin,
        )

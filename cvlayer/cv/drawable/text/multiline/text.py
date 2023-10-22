# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.drawable.defaults import (
    DEFAULT_FONT_FACE,
    DEFAULT_FONT_SCALE,
    DEFAULT_LINE_TYPE,
    DEFAULT_THICKNESS,
    MULTILINE_COLOR,
    MULTILINE_LINE_SPACING,
    MULTILINE_LINEFEED,
)
from cvlayer.cv.drawable.text.multiline.lines import draw_multiline_text_lines_coord
from cvlayer.cv.drawable.text.multiline.measure import measure_multiline_text_box_size
from cvlayer.typing import Number, PointN


def draw_multiline_text_coord(
    image: NDArray,
    text: str,
    x: Number,
    y: Number,
    font=DEFAULT_FONT_FACE,
    scale=DEFAULT_FONT_SCALE,
    color=MULTILINE_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    linefeed=MULTILINE_LINEFEED,
    spacing=MULTILINE_LINE_SPACING,
) -> NDArray:
    width, height, lines = measure_multiline_text_box_size(
        text,
        font,
        scale,
        thickness,
        linefeed,
        spacing,
    )
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
    )


def draw_multiline_text(
    image: NDArray,
    text: str,
    pos: PointN,
    font=DEFAULT_FONT_FACE,
    scale=DEFAULT_FONT_SCALE,
    color=MULTILINE_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    linefeed=MULTILINE_LINEFEED,
    spacing=MULTILINE_LINE_SPACING,
) -> NDArray:
    return draw_multiline_text_coord(
        image,
        text,
        pos[0],
        pos[1],
        font,
        scale,
        color,
        thickness,
        line,
        linefeed,
        spacing,
    )


class CvlDrawableTextMultilineText:
    @staticmethod
    def cvl_draw_multiline_text_coord(
        image: NDArray,
        text: str,
        x: Number,
        y: Number,
        font=DEFAULT_FONT_FACE,
        scale=DEFAULT_FONT_SCALE,
        color=MULTILINE_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        linefeed=MULTILINE_LINEFEED,
        spacing=MULTILINE_LINE_SPACING,
    ):
        return draw_multiline_text_coord(
            image,
            text,
            x,
            y,
            font,
            scale,
            color,
            thickness,
            line,
            linefeed,
            spacing,
        )

    @staticmethod
    def cvl_draw_multiline_text(
        image: NDArray,
        text: str,
        pos: PointN,
        font=DEFAULT_FONT_FACE,
        scale=DEFAULT_FONT_SCALE,
        color=MULTILINE_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        linefeed=MULTILINE_LINEFEED,
        spacing=MULTILINE_LINE_SPACING,
    ):
        return draw_multiline_text(
            image,
            text,
            pos,
            font,
            scale,
            color,
            thickness,
            line,
            linefeed,
            spacing,
        )

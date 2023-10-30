# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.drawable.defaults import (
    DEFAULT_FONT_COLOR,
    DEFAULT_FONT_FACE,
    DEFAULT_FONT_OUTLINE_COLOR,
    DEFAULT_FONT_OUTLINE_THICKNESS,
    DEFAULT_FONT_SCALE,
    DEFAULT_LINE_TYPE,
    DEFAULT_TEXT_ORIGIN,
    DEFAULT_THICKNESS,
)
from cvlayer.cv.drawable.text.text import draw_text_coord
from cvlayer.typing import Number, PointN


def draw_outline_text_coord(
    image: NDArray,
    text: str,
    x: Number,
    y: Number,
    font=DEFAULT_FONT_FACE,
    scale=DEFAULT_FONT_SCALE,
    color=DEFAULT_FONT_COLOR,
    outline_color=DEFAULT_FONT_OUTLINE_COLOR,
    thickness=DEFAULT_THICKNESS,
    outline_thickness=DEFAULT_FONT_OUTLINE_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    origin=DEFAULT_TEXT_ORIGIN,
) -> NDArray:
    draw_text_coord(
        image,
        text,
        x,
        y,
        font,
        scale,
        outline_color,
        thickness + outline_thickness,
        line,
        origin,
    )
    draw_text_coord(
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
    return image


def draw_outline_text(
    image: NDArray,
    text: str,
    pos: PointN,
    font=DEFAULT_FONT_FACE,
    scale=DEFAULT_FONT_SCALE,
    color=DEFAULT_FONT_COLOR,
    outline_color=DEFAULT_FONT_OUTLINE_COLOR,
    thickness=DEFAULT_THICKNESS,
    outline_thickness=DEFAULT_FONT_OUTLINE_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    origin=DEFAULT_TEXT_ORIGIN,
) -> NDArray:
    return draw_outline_text_coord(
        image,
        text,
        pos[0],
        pos[1],
        font,
        scale,
        color,
        outline_color,
        thickness,
        outline_thickness,
        line,
        origin,
    )


class CvlDrawableTextOutline:
    @staticmethod
    def cvl_draw_outline_text_coord(
        image: NDArray,
        text: str,
        x: Number,
        y: Number,
        font=DEFAULT_FONT_FACE,
        scale=DEFAULT_FONT_SCALE,
        color=DEFAULT_FONT_COLOR,
        outline_color=DEFAULT_FONT_OUTLINE_COLOR,
        thickness=DEFAULT_THICKNESS,
        outline_thickness=DEFAULT_FONT_OUTLINE_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        origin=DEFAULT_TEXT_ORIGIN,
    ):
        return draw_outline_text_coord(
            image,
            text,
            x,
            y,
            font,
            scale,
            color,
            outline_color,
            thickness,
            outline_thickness,
            line,
            origin,
        )

    @staticmethod
    def cvl_draw_outline_text(
        image: NDArray,
        text: str,
        pos: PointN,
        font=DEFAULT_FONT_FACE,
        scale=DEFAULT_FONT_SCALE,
        color=DEFAULT_FONT_COLOR,
        outline_color=DEFAULT_FONT_OUTLINE_COLOR,
        thickness=DEFAULT_THICKNESS,
        outline_thickness=DEFAULT_FONT_OUTLINE_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        origin=DEFAULT_TEXT_ORIGIN,
    ) -> NDArray:
        return draw_outline_text(
            image,
            text,
            pos,
            font,
            scale,
            color,
            outline_color,
            thickness,
            outline_thickness,
            line,
            origin,
        )

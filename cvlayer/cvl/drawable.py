# -*- coding: utf-8 -*-

from typing import Optional, Tuple

from cvlayer.cv.drawable import (
    COLOR,
    FILLED,
    FONT,
    FONT_SCALE,
    LINE_AA,
    LINE_TYPE,
    MULTILINE_BACKGROUND_ALPHA,
    MULTILINE_BACKGROUND_COLOR,
    MULTILINE_COLOR,
    MULTILINE_LINEFEED,
    MULTILINE_MARGIN,
    OUTLINE_COLOR,
    OUTLINE_FILL_COLOR,
    OUTLINE_THICKNESS,
    RADIUS,
    THICKNESS,
    draw_circle,
    draw_line,
    draw_multiline_text,
    draw_outline_text,
    draw_point,
    draw_rectangle,
    measure_multiline_text_box_size,
)
from cvlayer.types import Image, Number, PointT, RectT


class CvlDrawable:
    @staticmethod
    def cvl_draw_point(
        image: Image,
        x: Number,
        y: Number,
        radius=RADIUS,
        color=COLOR,
        thickness=FILLED,
        line_type=LINE_AA,
    ) -> None:
        draw_point(image, x, y, radius, color, thickness, line_type)

    @staticmethod
    def cvl_draw_line(
        image: Image,
        point1: PointT,
        point2: PointT,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
    ) -> None:
        draw_line(image, point1, point2, color, thickness, line_type)

    @staticmethod
    def cvl_draw_rectangle(
        image: Image,
        roi: RectT,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
    ) -> None:
        draw_rectangle(image, roi, color, thickness, line_type)

    @staticmethod
    def cvl_draw_circle(
        image: Image,
        x: Number,
        y: Number,
        radius=RADIUS,
        color=COLOR,
        thickness=FILLED,
        line_type=LINE_TYPE,
    ) -> None:
        draw_circle(image, x, y, radius, color, thickness, line_type)

    @staticmethod
    def cvl_draw_outline_text(
        image: Image,
        text: str,
        x: Number,
        y: Number,
        font=FONT,
        scale=FONT_SCALE,
        fill_color=OUTLINE_FILL_COLOR,
        outline_color=OUTLINE_COLOR,
        thickness=THICKNESS,
        outline_thickness=OUTLINE_THICKNESS,
        line_type=LINE_TYPE,
    ) -> None:
        draw_outline_text(
            image,
            text,
            x,
            y,
            font,
            scale,
            fill_color,
            outline_color,
            thickness,
            outline_thickness,
            line_type,
        )

    @staticmethod
    def cvl_measure_multiline_text_box_size(
        text: str,
        font=FONT,
        scale=FONT_SCALE,
        thickness=THICKNESS,
        linefeed=MULTILINE_LINEFEED,
    ) -> Tuple[int, int]:
        return measure_multiline_text_box_size(text, font, scale, thickness, linefeed)

    @staticmethod
    def cvl_draw_multiline_text(
        image: Image,
        text: str,
        x: Number,
        y: Number,
        font=FONT,
        scale=FONT_SCALE,
        color=MULTILINE_COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
        linefeed=MULTILINE_LINEFEED,
        background_color=MULTILINE_BACKGROUND_COLOR,
        background_alpha=MULTILINE_BACKGROUND_ALPHA,
        margin=MULTILINE_MARGIN,
        canvas_width: Optional[int] = None,
        canvas_height: Optional[int] = None,
    ) -> None:
        draw_multiline_text(
            image,
            text,
            x,
            y,
            font,
            scale,
            color,
            thickness,
            line_type,
            linefeed,
            background_color,
            background_alpha,
            margin,
            canvas_width,
            canvas_height,
        )

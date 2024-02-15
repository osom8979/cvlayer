# -*- coding: utf-8 -*-

from typing import Tuple

import cv2
from numpy import full, uint8
from numpy.typing import NDArray

from cvlayer.cv.drawable.defaults import (
    DEFAULT_FONT_FACE,
    DEFAULT_FONT_SCALE,
    DEFAULT_LINE_TYPE,
    DEFAULT_THICKNESS,
    MULTILINE_BACKGROUND_ALPHA,
    MULTILINE_BACKGROUND_COLOR,
    MULTILINE_BOX_ANCHOR,
    MULTILINE_BOX_ANCHOR_X,
    MULTILINE_BOX_ANCHOR_Y,
    MULTILINE_BOX_MARGIN,
    MULTILINE_COLOR,
    MULTILINE_LINE_SPACING,
    MULTILINE_LINEFEED,
)
from cvlayer.cv.drawable.text.multiline.lines import draw_multiline_text_lines_coord
from cvlayer.cv.drawable.text.multiline.measure import measure_multiline_text_box_size
from cvlayer.typing import Number, PointN, RectI


def draw_multiline_text_box_coord(
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
    background_color=MULTILINE_BACKGROUND_COLOR,
    background_alpha=MULTILINE_BACKGROUND_ALPHA,
    margin=MULTILINE_BOX_MARGIN,
    anchor_x=MULTILINE_BOX_ANCHOR_X,
    anchor_y=MULTILINE_BOX_ANCHOR_Y,
) -> Tuple[NDArray, RectI]:
    assert 0 <= anchor_x <= 1
    assert 0 <= anchor_y <= 1
    assert 0 <= background_alpha <= 1

    ch = image.shape[0]
    cw = image.shape[1]

    bw, bh, lines = measure_multiline_text_box_size(
        text, font, scale, thickness, linefeed, spacing
    )
    bw += margin * 2
    bh += margin * 2
    box = full((bh, bw, 3), background_color, dtype=uint8)

    bx = bw * anchor_x
    by = bh * anchor_y
    x1 = max(int((x + cw * anchor_x) - bx), 0)
    y1 = max(int((y + ch * anchor_y) - by), 0)
    x2 = min(x1 + bw, cw)
    y2 = min(y1 + bh, ch)
    w = x2 - x1
    h = y2 - y1

    assert 0 <= x1 <= cw
    assert 0 <= y1 <= ch
    assert 0 <= x2 <= cw
    assert 0 <= y2 <= ch
    assert w <= bw
    assert h <= bh

    img_area = image[y1:y2, x1:x2]
    box_area = box[0:h, 0:w]

    alpha = background_alpha
    beta = 1.0 - background_alpha

    mixed: NDArray
    if alpha >= 1:
        mixed = box_area
    elif beta >= 1:
        mixed = img_area
    else:
        mixed = cv2.addWeighted(box_area, alpha, img_area, beta, 0)

    draw_multiline_text_lines_coord(
        mixed,
        lines,
        x + margin,
        y + margin,
        font,
        scale,
        color,
        thickness,
        line,
        spacing,
    )
    image[y1:y2, x1:x2] = mixed
    roi = x1, y1, x2, y2
    return image, roi


def draw_multiline_text_box(
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
    background_color=MULTILINE_BACKGROUND_COLOR,
    background_alpha=MULTILINE_BACKGROUND_ALPHA,
    margin=MULTILINE_BOX_MARGIN,
    anchor=MULTILINE_BOX_ANCHOR,
) -> Tuple[NDArray, RectI]:
    return draw_multiline_text_box_coord(
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
        background_color,
        background_alpha,
        margin,
        anchor[0],
        anchor[1],
    )


class CvlDrawableTextMultilineBox:
    @staticmethod
    def cvl_draw_multiline_text_box_coord(
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
        background_color=MULTILINE_BACKGROUND_COLOR,
        background_alpha=MULTILINE_BACKGROUND_ALPHA,
        margin=MULTILINE_BOX_MARGIN,
        anchor_x=MULTILINE_BOX_ANCHOR_X,
        anchor_y=MULTILINE_BOX_ANCHOR_Y,
    ):
        return draw_multiline_text_box_coord(
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
            background_color,
            background_alpha,
            margin,
            anchor_x,
            anchor_y,
        )

    @staticmethod
    def cvl_draw_multiline_text_box(
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
        background_color=MULTILINE_BACKGROUND_COLOR,
        background_alpha=MULTILINE_BACKGROUND_ALPHA,
        margin=MULTILINE_BOX_MARGIN,
        anchor=MULTILINE_BOX_ANCHOR,
    ):
        return draw_multiline_text_box(
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
            background_color,
            background_alpha,
            margin,
            anchor,
        )

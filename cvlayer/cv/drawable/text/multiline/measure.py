# -*- coding: utf-8 -*-

from typing import List, NamedTuple

import cv2

from cvlayer.cv.drawable.defaults import (
    DEFAULT_FONT_FACE,
    DEFAULT_FONT_SCALE,
    DEFAULT_THICKNESS,
    MULTILINE_LINE_SPACING,
    MULTILINE_LINEFEED,
)
from cvlayer.cv.types.font_face import normalize_font_face
from cvlayer.typing import SizeI


class LineTextSize(NamedTuple):
    text: str
    size: SizeI
    baseline: int


class MultilineTextBoxSize(NamedTuple):
    box_width: int
    box_height: int
    lines: List[LineTextSize]


def measure_multiline_text_box_size(
    text: str,
    font=DEFAULT_FONT_FACE,
    scale=DEFAULT_FONT_SCALE,
    thickness=DEFAULT_THICKNESS,
    linefeed=MULTILINE_LINEFEED,
    spacing=MULTILINE_LINE_SPACING,
) -> MultilineTextBoxSize:
    tws = list()
    ths = list()
    lines = list()
    _font = normalize_font_face(font)
    for line in text.split(linefeed):
        text_size = cv2.getTextSize(line, _font, scale, thickness)
        text_width, text_height = text_size[0]
        baseline = text_size[1]
        line_height = text_height + baseline + spacing
        tws.append(text_width)
        ths.append(line_height)
        lines.append(LineTextSize(line, (text_width, text_height), baseline))
    box_width = max(tws)
    box_height = sum(ths)
    if len(ths) >= 1:
        box_height -= spacing  # The last line has no bottom line spacing.
    return MultilineTextBoxSize(box_width, box_height, lines)


class CvlDrawableTextMultilineMeasure:
    @staticmethod
    def cvl_measure_multiline_text_box_size(
        text: str,
        font=DEFAULT_FONT_FACE,
        scale=DEFAULT_FONT_SCALE,
        thickness=DEFAULT_THICKNESS,
        linefeed=MULTILINE_LINEFEED,
        line_spacing=MULTILINE_LINE_SPACING,
    ):
        return measure_multiline_text_box_size(
            text,
            font,
            scale,
            thickness,
            linefeed,
            line_spacing,
        )

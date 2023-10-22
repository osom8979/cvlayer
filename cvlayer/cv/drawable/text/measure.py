# -*- coding: utf-8 -*-

from typing import NamedTuple

import cv2

from cvlayer.cv.drawable.defaults import (
    DEFAULT_FONT_FACE,
    DEFAULT_FONT_SCALE,
    DEFAULT_THICKNESS,
)
from cvlayer.cv.types.font_face import normalize_font_face
from cvlayer.typing import SizeI


class TextSize(NamedTuple):
    size: SizeI
    baseline: int


def get_font_scale_from_height(
    pixel_height: int,
    font=DEFAULT_FONT_FACE,
    thickness=DEFAULT_THICKNESS,
) -> float:
    _font = normalize_font_face(font)
    return cv2.getFontScaleFromHeight(_font, pixel_height, thickness)


def get_text_size(
    text: str,
    font=DEFAULT_FONT_FACE,
    scale=DEFAULT_FONT_SCALE,
    thickness=DEFAULT_THICKNESS,
) -> TextSize:
    _font = normalize_font_face(font)
    text_size = cv2.getTextSize(text, _font, scale, thickness)
    width, height = text_size[0]
    baseline = text_size[1]
    return TextSize((width, height), baseline)


class CvlDrawableTextMeasure:
    @staticmethod
    def cvl_get_font_scale_from_height(
        pixel_height: int,
        font=DEFAULT_FONT_FACE,
        thickness=DEFAULT_THICKNESS,
    ):
        return get_font_scale_from_height(font, pixel_height, thickness)

    @staticmethod
    def cvl_get_text_size(
        text: str,
        font=DEFAULT_FONT_FACE,
        scale=DEFAULT_FONT_SCALE,
        thickness=DEFAULT_THICKNESS,
    ):
        return get_text_size(text, font, scale, thickness)

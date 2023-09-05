# -*- coding: utf-8 -*-

from numpy import full, ndarray, uint8, zeros

from cvlayer.typing import Color, Image, ImageShape


def make_image_with_shape(shape: ImageShape, data: bytes) -> Image:
    return ndarray(shape, dtype=uint8, buffer=data)


def make_image(width: int, height: int, data: bytes) -> Image:
    return make_image_with_shape((height, width, 3), data)


def make_image_filled(width: int, height: int, color: Color) -> Image:
    return full((height, width, 3), color, dtype=uint8)


def make_image_empty(width: int, height: int) -> Image:
    return zeros((height, width, 3), dtype=uint8)

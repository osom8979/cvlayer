# -*- coding: utf-8 -*-

from numpy import full, ndarray, uint8, zeros
from numpy.random import randint
from numpy.typing import NDArray

from cvlayer.cv.types.color import ColorLike, normalize_color


def make_image(width: int, height: int, data: bytes) -> NDArray[uint8]:
    return ndarray((height, width, 3), dtype=uint8, buffer=data)


def make_image_filled(width: int, height: int, color: ColorLike) -> NDArray[uint8]:
    return full((height, width, 3), normalize_color(color), dtype=uint8)


def make_image_empty(width: int, height: int) -> NDArray[uint8]:
    return zeros((height, width, 3), dtype=uint8)


def make_image_random(width: int, height: int) -> NDArray[uint8]:
    return randint(low=0, high=256, size=(height, width, 3), dtype=uint8)

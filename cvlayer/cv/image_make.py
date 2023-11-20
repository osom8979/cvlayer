# -*- coding: utf-8 -*-

from cvlayer.cv.types.color import Color
from cvlayer.np.image import (
    make_image,
    make_image_empty,
    make_image_filled,
    make_image_random,
)


class CvlImageMake:
    @staticmethod
    def cvl_make_image(width: int, height: int, data: bytes):
        return make_image(width, height, data)

    @staticmethod
    def cvl_make_image_filled(width: int, height: int, color: Color):
        return make_image_filled(width, height, color)

    @staticmethod
    def cvl_make_image_empty(width: int, height: int):
        return make_image_empty(width, height)

    @staticmethod
    def cvl_make_image_random(width: int, height: int):
        return make_image_random(width, height)

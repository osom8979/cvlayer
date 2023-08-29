# -*- coding: utf-8 -*-

from cvlayer.np.image import make_image, make_image_filled, make_image_with_shape
from cvlayer.types import Color, Image, ImageShape


class CvlMakeImage:
    @staticmethod
    def cvl_make_image_with_shape(shape: ImageShape, data: bytes) -> Image:
        return make_image_with_shape(shape, data)

    @staticmethod
    def cvl_make_image(width: int, height: int, data: bytes) -> Image:
        return make_image(width, height, data)

    @staticmethod
    def cvl_make_image_filled(width: int, height: int, color: Color) -> Image:
        return make_image_filled(width, height, color)

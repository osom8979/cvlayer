# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.cv.image import DEFAULT_INTERPOLATION, Interpolation, scale_image
from cvlayer.np.image import make_image, make_image_filled, make_image_with_shape
from cvlayer.types import Color, Image, ImageShape


class CvlImage:
    InterpolationType = Interpolation

    @staticmethod
    def cvl_make_image_with_shape(shape: ImageShape, data: bytes) -> Image:
        return make_image_with_shape(shape, data)

    @staticmethod
    def cvl_make_image(width: int, height: int, data: bytes) -> Image:
        return make_image(width, height, data)

    @staticmethod
    def cvl_make_image_filled(width: int, height: int, color: Color) -> Image:
        return make_image_filled(width, height, color)

    @staticmethod
    def cvl_scale_image(
        src: NDArray,
        scale: float,
        interpolation=DEFAULT_INTERPOLATION,
    ) -> NDArray:
        return scale_image(src, scale, interpolation)

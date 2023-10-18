# -*- coding: utf-8 -*-

from numpy import float16, float32, float64, uint8
from numpy.typing import NDArray


def image_normalize_float32_as_uint8(src: NDArray) -> NDArray:
    assert src.dtype in (float16, float32, float64)
    assert src.min() >= 0.0
    assert src.max() <= 1.0
    return (src * 255).astype(uint8)


class CvlImageNormalize:
    @staticmethod
    def cvl_image_normalize_float32_as_uint8(src: NDArray):
        return image_normalize_float32_as_uint8(src)

# -*- coding: utf-8 -*-

from cvlayer.cv.blur import (
    DEFAULT_BILATERAL_FILTER_SIGMA_COLOR,
    DEFAULT_BILATERAL_FILTER_SIGMA_SPACE,
    DEFAULT_GAUSSIAN_BLUR_SIGMA_X,
    DEFAULT_GAUSSIAN_BLUR_SIGMA_Y,
    DEFAULT_KSIZE,
    bilateral_filter,
    gaussian_blur,
)
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmBlur(LayerManagerMixinBase):
    def cvm_bilateral_filter(
        self,
        name: str,
        d=9,
        sigma_color=DEFAULT_BILATERAL_FILTER_SIGMA_COLOR,
        sigma_space=DEFAULT_BILATERAL_FILTER_SIGMA_SPACE,
    ):
        with self.layer(name) as layer:
            d = layer.param("d").build_uint(d, 1).value
            sc = layer.param("sc").build_float(sigma_color, 0.1, step=0.1).value
            ss = layer.param("ss").build_float(sigma_space, 0.1, step=0.1).value
            result = bilateral_filter(layer.prev_frame, d, sc, ss)
            layer.frame = result
        return result

    def cvm_gaussian_blur(
        self,
        name: str,
        ksize=DEFAULT_KSIZE,
        sigma_x=DEFAULT_GAUSSIAN_BLUR_SIGMA_X,
        sigma_y=DEFAULT_GAUSSIAN_BLUR_SIGMA_Y,
    ):
        with self.layer(name) as layer:
            kx = layer.param("kx").build_uint(ksize[0], 1, step=2).value
            ky = layer.param("ky").build_uint(ksize[1], 1, step=2).value
            sx = layer.param("sx").build_float(sigma_x, 0.0, step=0.1).value
            sy = layer.param("sy").build_float(sigma_y, 0.0, step=0.1).value
            result = gaussian_blur(layer.prev_frame, (kx, ky), sx, sy)
            layer.frame = result
        return result

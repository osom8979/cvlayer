# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.blur import bilateral_filter, gaussian_blur
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmBlur(LayerManagerMixinBase):
    def cvm_bilateral_filter(
        self,
        name: str,
        d=9,
        sigma_color=75.0,
        sigma_space=75.0,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            d = layer.param("d").build_uint(d, 1).value
            sc = layer.param("sc").build_float(sigma_color, 0.1, step=0.1).value
            ss = layer.param("ss").build_float(sigma_space, 0.1, step=0.1).value
            src = frame if frame is not None else layer.prev_frame
            result = bilateral_filter(src, d, sc, ss)
            layer.frame = result
        return result

    def cvm_gaussian_blur(
        self,
        name: str,
        ksize=(3, 3),
        sigma_x=0.0,
        sigma_y=0.0,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            kx = layer.param("kx").build_uint(ksize[0], 1, step=2).value
            ky = layer.param("ky").build_uint(ksize[1], 1, step=2).value
            sx = layer.param("sx").build_float(sigma_x, 0.0, step=0.1).value
            sy = layer.param("sy").build_float(sigma_y, 0.0, step=0.1).value
            src = frame if frame is not None else layer.prev_frame
            result = gaussian_blur(src, (kx, ky), sx, sy)
            layer.frame = result
        return result

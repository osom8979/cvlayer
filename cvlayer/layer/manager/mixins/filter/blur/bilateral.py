# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.filter.blur.bilateral import bilateral_filter
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmFilterBlurBilateral(LayerManagerMixinBase):
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

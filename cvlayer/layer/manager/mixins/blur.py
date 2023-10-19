# -*- coding: utf-8 -*-

from cvlayer.cv.blur import (
    DEFAULT_BILATERAL_FILTER_SIGMA_COLOR,
    DEFAULT_BILATERAL_FILTER_SIGMA_SPACE,
    bilateral_filter,
)
from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase


class CvmBlur(_LayerManagerMixinBase):
    def cvm_bilateral_filter(
        self,
        name: str,
        d=9,
        sigma_color=DEFAULT_BILATERAL_FILTER_SIGMA_COLOR,
        sigma_space=DEFAULT_BILATERAL_FILTER_SIGMA_SPACE,
    ):
        with self.layer(name) as layer:
            d = layer.param("d").build_unsigned(d, 1).value
            sc = layer.param("sc").build_floating(sigma_color, 0.1, step=0.1).value
            ss = layer.param("ss").build_floating(sigma_space, 0.1, step=0.1).value
            result = bilateral_filter(layer.prev_frame, d, sc, ss)
            layer.frame = result
        return result

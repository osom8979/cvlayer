# -*- coding: utf-8 -*-

from cvlayer.cv.threshold import PIXEL_8BIT_MAX, ThresholdMethod, threshold_otsu
from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase


class CvmThreshold(_LayerManagerMixinBase):
    def cvm_threshold_otsu(
        self,
        name: str,
        max_value=PIXEL_8BIT_MAX,
        method=ThresholdMethod.BINARY,
    ):
        with self.layer(name) as layer:
            mv = layer.param("max").build_unsigned(max_value).value
            m = layer.param("method").build_enumeration(method).value
            result = threshold_otsu(layer.prev_frame, mv, m).threshold_image
            layer.frame = result
        return result

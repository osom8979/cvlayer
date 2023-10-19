# -*- coding: utf-8 -*-

from cvlayer.cv.canny import CANNY_THRESHOLD_MAX, CANNY_THRESHOLD_MIN, canny
from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase


class CvmCanny(_LayerManagerMixinBase):
    def cvm_canny(
        self,
        name: str,
        th_min=CANNY_THRESHOLD_MIN,
        th_max=CANNY_THRESHOLD_MAX,
    ):
        with self.layer(name) as layer:
            th_min = layer.param("th_min").build_unsigned(th_min).value
            th_max = layer.param("th_max").build_unsigned(th_max).value
            result = canny(layer.prev_frame, th_min, th_max)
            layer.frame = result
        return result

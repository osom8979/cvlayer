# -*- coding: utf-8 -*-

from cvlayer.cv.canny import CANNY_THRESHOLD_MAX, CANNY_THRESHOLD_MIN, canny
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmCanny(LayerManagerMixinBase):
    def cvm_canny(
        self,
        name: str,
        th_min=CANNY_THRESHOLD_MIN,
        th_max=CANNY_THRESHOLD_MAX,
    ):
        with self.layer(name) as layer:
            th_min = layer.param("th_min").build_uint(th_min).value
            th_max = layer.param("th_max").build_uint(th_max).value
            result = canny(layer.prev_frame, th_min, th_max)
            layer.frame = result
        return result

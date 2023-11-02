# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.filter.edge.canny import canny
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmFilterEdgeCanny(LayerManagerMixinBase):
    def cvm_canny(
        self,
        name: str,
        th_min=30,
        th_max=70,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            th_min = layer.param("th_min").build_uint(th_min).value
            th_max = layer.param("th_max").build_uint(th_max).value
            src = frame if frame is not None else layer.prev_frame
            result = canny(src, th_min, th_max)
            layer.frame = result
        return result

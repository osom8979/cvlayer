# -*- coding: utf-8 -*-

from cvlayer.cv.histogram import equalize_hist
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmHistogram(LayerManagerMixinBase):
    def cvm_equalize_hist(self, name: str):
        with self.layer(name) as layer:
            layer.frame = result = equalize_hist(layer.prev_frame)
        return result

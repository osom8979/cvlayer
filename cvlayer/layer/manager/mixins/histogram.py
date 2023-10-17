# -*- coding: utf-8 -*-

from cvlayer.cv.histogram import equalize_hist
from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase


class CvmHistogram(_LayerManagerMixinBase):
    def cvm_equalize_hist(self):
        with self.layer("cvm_equalize_hist") as layer:
            result = equalize_hist(layer.prev_frame)
            layer.frame = result
        return result

# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.histogram import equalize_hist
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmHistogram(LayerManagerMixinBase):
    def cvm_equalize_hist(self, name: str, frame: Optional[NDArray] = None):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            layer.frame = result = equalize_hist(src)
        return result

# -*- coding: utf-8 -*-

from typing import Optional

from numpy import uint8
from numpy.typing import NDArray

from cvlayer.cv.basic import channel_mean_abs_diff
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmBasic(LayerManagerMixinBase):
    def cvm_channel_mean_abs_diff(self, name: str, frame: Optional[NDArray] = None):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            result = channel_mean_abs_diff(src).astype(uint8)
            layer.frame = result
        return result

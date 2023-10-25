# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.bitwise import bitwise_not
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmBitwise(LayerManagerMixinBase):
    def cvm_bitwise_not(
        self,
        name: str,
        mask: Optional[NDArray] = None,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            result = bitwise_not(src, mask)
            layer.frame = result
        return result

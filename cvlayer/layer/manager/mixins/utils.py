# -*- coding: utf-8 -*-

from typing import Any, Optional

from numpy.typing import NDArray

from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmUtils(LayerManagerMixinBase):
    def cvm_select(self, name: str, src: NDArray, data: Optional[Any] = None):
        with self.layer(name) as layer:
            layer.frame = src
            layer.data = data
        return src, data

    def cvm_select_roi(self, name: str, frame: Optional[NDArray] = None):
        with self.layer(name) as layer:
            roi = layer.param("roi").build_select_roi().value
            self.set_roi(roi)
            layer.frame = frame if frame is not None else layer.prev_frame
            layer.data = roi
        return roi

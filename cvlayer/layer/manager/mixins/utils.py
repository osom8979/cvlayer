# -*- coding: utf-8 -*-

from typing import Any, Optional

from numpy.typing import NDArray

from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase


class CvmUtils(_LayerManagerMixinBase):
    def cvm_select(self, src: NDArray, data: Optional[Any] = None):
        with self.layer("cvm_select") as layer:
            layer.frame = src
            layer.data = data
        return src, data

    def cvm_select_roi(self):
        with self.layer("cvm_select_roi") as layer:
            roi = layer.param("roi").build_select_roi().value
            self.set_roi(roi)
            layer.frame = layer.prev_frame
            layer.data = roi
        return roi

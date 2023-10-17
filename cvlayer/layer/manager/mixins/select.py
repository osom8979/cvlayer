# -*- coding: utf-8 -*-

from typing import Any, Optional

from numpy.typing import NDArray

from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase


class CvmSelect(_LayerManagerMixinBase):
    def cvm_select_image(self, src: NDArray, data: Optional[Any] = None):
        with self.layer("cvm_select_image") as layer:
            layer.frame = src
            layer.data = data
        return src

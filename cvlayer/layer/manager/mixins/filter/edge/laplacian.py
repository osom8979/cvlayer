# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.filter.edge.laplacian import laplacian
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmFilterEdgeLaplacian(LayerManagerMixinBase):
    def cvm_laplacian(
        self,
        name: str,
        kernel_size=1,
        scale=1.0,
        delta=0.0,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            ksize = layer.param("kernel_size").build_uint(kernel_size, 1, step=2).value
            s = layer.param("scale").build_float(scale, 0.0, step=0.1).value
            d = layer.param("delta").build_float(delta, 0.0, step=0.1).value
            src = frame if frame is not None else layer.prev_frame
            result = laplacian(src, kernel_size=ksize, scale=s, delta=d)
            layer.frame = result
        return result

# -*- coding: utf-8 -*-

from typing import Final, Optional, Sequence

from numpy import clip, uint8
from numpy.typing import NDArray

from cvlayer.cv.filter.edge.sobel import (
    AVAILABLE_KERNEL_SIZE,
    DEFAULT_DELTA,
    DEFAULT_DX,
    DEFAULT_DY,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_SCALE,
    sobel_magnitude,
)
from cvlayer.cv.types.border import (
    DEFAULT_BORDER_TYPE,
    BorderType,
    normalize_border_type,
)
from cvlayer.cv.types.data_type import CV_32F
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase

_KSIZE_LIST: Final[Sequence[int]] = AVAILABLE_KERNEL_SIZE


class CvmFilterEdgeSobelMagnitude(LayerManagerMixinBase):
    def cvm_sobel_magnitude(
        self,
        name: str,
        dx=DEFAULT_DX,
        dy=DEFAULT_DY,
        kernel_size=DEFAULT_KERNEL_SIZE,
        scale=DEFAULT_SCALE,
        delta=DEFAULT_DELTA,
        border=DEFAULT_BORDER_TYPE,
        use_clip=True,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame

            init_border = BorderType(normalize_border_type(border))

            _dx = layer.param("dx").build_int(dx).value
            _dy = layer.param("dy").build_int(dy).value
            _ksize = layer.param("ksize").build_list(_KSIZE_LIST, kernel_size).value
            _scale = layer.param("scale").build_float(scale).value
            _delta = layer.param("delta").build_float(delta).value
            _border = layer.param("border").build_enum(init_border).value
            _use_clip = layer.param("use_clip").build_bool(use_clip).value

            m = sobel_magnitude(
                frame=src,
                ddepth=CV_32F,
                dx=_dx,
                dy=_dy,
                kernel_size=_ksize,
                scale=_scale,
                delta=_delta,
                border=_border,
            )

            if _use_clip:
                layer.frame = clip(m, 0, 255).astype(uint8)
            else:
                layer.frame = (m / m.max() * 255).astype(uint8)

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
    sobel_cart_to_polar,
)
from cvlayer.cv.types.angle import DEFAULT_ANGLE_TYPE, AngleType, normalize_angle_type
from cvlayer.cv.types.border import (
    DEFAULT_BORDER_TYPE,
    BorderType,
    normalize_border_type,
)
from cvlayer.cv.types.data_type import CV_32F
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase

_KSIZE_LIST: Final[Sequence[int]] = AVAILABLE_KERNEL_SIZE


class CvmFilterEdgeSobelCartToPolar(LayerManagerMixinBase):
    def cvm_sobel_cart_to_polar(
        self,
        name: str,
        dx=DEFAULT_DX,
        dy=DEFAULT_DY,
        kernel_size=DEFAULT_KERNEL_SIZE,
        scale=DEFAULT_SCALE,
        delta=DEFAULT_DELTA,
        border=DEFAULT_BORDER_TYPE,
        angle_in_degrees=DEFAULT_ANGLE_TYPE,
        use_clip=True,
        show_magnitude=True,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame

            init_border = BorderType(normalize_border_type(border))
            init_angle = AngleType(normalize_angle_type(angle_in_degrees))

            _dx = layer.param("dx").build_int(dx).value
            _dy = layer.param("dy").build_int(dy).value
            _ksize = layer.param("ksize").build_list(_KSIZE_LIST, kernel_size).value
            _scale = layer.param("scale").build_float(scale).value
            _delta = layer.param("delta").build_float(delta).value
            _border = layer.param("border").build_enum(init_border).value
            _angle = layer.param("angle").build_enum(init_angle).value
            _use_clip = layer.param("use_clip").build_bool(use_clip).value
            _sm = layer.param("show_magnitude").build_bool(show_magnitude).value

            m, p = sobel_cart_to_polar(
                frame=src,
                ddepth=CV_32F,
                dx=_dx,
                dy=_dy,
                kernel_size=_ksize,
                scale=_scale,
                delta=_delta,
                border=_border,
                angle_in_degrees=_angle,
            )

            if _use_clip:
                magnitude_mask = clip(m, 0, 255).astype(uint8)
                phase_mask = clip(p, 0, 255).astype(uint8)
            else:
                magnitude_mask = (m / m.max() * 255).astype(uint8)
                phase_mask = (p / p.max() * 255).astype(uint8)

            layer.frame = magnitude_mask if _sm else phase_mask

        return magnitude_mask, phase_mask

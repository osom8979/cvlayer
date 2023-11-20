# -*- coding: utf-8 -*-

from typing import Optional, Sequence

from numpy.typing import NDArray

from cvlayer.cv.border import copy_make_border
from cvlayer.cv.types.border import (
    DEFAULT_BORDER_TYPE,
    BorderType,
    normalize_border_type,
)
from cvlayer.cv.types.color import ColorLike, normalize_color
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmBorder(LayerManagerMixinBase):
    def cvm_copy_make_border(
        self,
        name: str,
        top=1,
        bottom=1,
        left=1,
        right=1,
        border=DEFAULT_BORDER_TYPE,
        value: Optional[ColorLike] = None,
        isolated=False,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame

            init_border = BorderType(normalize_border_type(border))
            exc_border = (BorderType.TRANSPARENT, BorderType.ISOLATED)
            init_value = normalize_color(value if value is not None else 0.0)
            init_b = init_value[0] if len(init_value) >= 1 else 0.0
            init_g = init_value[1] if len(init_value) >= 2 else 0.0
            init_r = init_value[2] if len(init_value) >= 3 else 0.0
            init_a = init_value[3] if len(init_value) >= 4 else 255.0

            _top = layer.param("top").build_uint(top).value
            _bottom = layer.param("bottom").build_uint(bottom).value
            _left = layer.param("left").build_uint(left).value
            _right = layer.param("right").build_uint(right).value
            _border = layer.param("border").build_enum(init_border, exc_border).value
            b = layer.param("b").build_float(init_b, 0.0, step=1.0).value
            g = layer.param("g").build_float(init_g, 0.0, step=1.0).value
            r = layer.param("r").build_float(init_r, 0.0, step=1.0).value
            a = layer.param("a").build_float(init_a, 0.0, step=1.0).value
            _isolated = layer.param("isolated").build_bool(isolated).value

            _value: Sequence[float]
            if len(src.shape) == 2:
                _value = (b,)
            elif len(src.shape) == 3:
                if src.shape[2] == 3:
                    _value = (b, g, r)
                elif src.shape[2] == 4:
                    _value = (b, g, r, a)
                else:
                    raise ValueError(f"Unsupported src.shape: {src.shape}")
            else:
                raise ValueError(f"Unsupported src.shape: {src.shape}")

            result = copy_make_border(
                src,
                _top,
                _bottom,
                _left,
                _right,
                _border,
                _value,
                _isolated,
            )
            layer.frame = result
        return result

# -*- coding: utf-8 -*-

from typing import Optional, Sequence

from numpy.typing import NDArray

from cvlayer.cv.drawable.point import draw_point
from cvlayer.cv.types.line_type import DEFAULT_LINE_TYPE, LineType, normalize_line_type
from cvlayer.cv.types.shape import PointI
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase
from cvlayer.palette.basic import RED


class CvmPerspectiveSelect(LayerManagerMixinBase):
    def cvm_perspective_select_points(
        self,
        name: str,
        points: Optional[Sequence[PointI]] = None,
        radius=4,
        color=RED,
        line=DEFAULT_LINE_TYPE,
        use_deepcopy=True,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            init_points = points if points else list()
            init_line = LineType(normalize_line_type(line))

            ps = layer.param("points").build_select_points(init_points).value
            rd = layer.param("radius").build_uint(radius, 1).value
            lt = layer.param("line").build_enum(init_line).value
            dc = layer.param("deepcopy").build_bool(use_deepcopy).value

            src = frame if frame is not None else layer.prev_frame
            canvas = src.copy() if dc else src
            for p in ps:
                draw_point(canvas, p, radius=rd, color=color, line=lt)

            layer.frame = canvas
            layer.data = ps
        return canvas, ps

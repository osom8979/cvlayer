# -*- coding: utf-8 -*-

from typing import Optional, Sequence

from numpy.typing import NDArray

from cvlayer.cv.perspective import (
    get_perspective_transform_with_quadrilateral,
    warp_perspective,
)
from cvlayer.cv.types.shape import PointI
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmPerspectiveTransform(LayerManagerMixinBase):
    def cvm_perspective_transform_with_quadrilateral(
        self,
        name: str,
        points: Optional[Sequence[PointI]] = None,
        sx=1.0,
        sy=1.0,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame else layer.prev_frame

            p0 = points[0] if points and len(points) >= 1 else (0, 0)
            p1 = points[1] if points and len(points) >= 2 else (0, 0)
            p2 = points[2] if points and len(points) >= 3 else (0, 0)
            p3 = points[3] if points and len(points) >= 4 else (0, 0)
            init_points = p0, p1, p2, p3

            ps = layer.param("points").build_select_points(init_points).value
            w = layer.param("sx").build_float(sx, 1.0, step=0.1).value
            h = layer.param("sy").build_float(sy, 1.0, step=0.1).value

            xs = list(map(lambda p: p[0], ps))
            ys = list(map(lambda p: p[1], ps))
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            width, height = abs(x2 - x1) * w, abs(y2 - y1) * h
            roi = 0, 0, width, height

            m = get_perspective_transform_with_quadrilateral(
                left_top=ps[0],
                left_bottom=ps[1],
                right_top=ps[2],
                right_bottom=ps[3],
                destination_roi=roi,
            )
            result = warp_perspective(src, matrix=m, dsize=(width, height))
            layer.frame = result
        return result

# -*- coding: utf-8 -*-

from typing import Dict, Optional

from numpy import ndarray, uint8, zeros
from numpy.typing import NDArray

from cvlayer.cv.contours import (
    FindContoursMethod,
    FindContoursMode,
    contour_area,
    find_contours,
)
from cvlayer.cv.contours_moments import moments
from cvlayer.cv.cvt_color import cvt_color_GRAY2BGR
from cvlayer.cv.drawable.contours import draw_contour
from cvlayer.cv.drawable.point import draw_point
from cvlayer.cv.drawable.text.outline import draw_outline_text
from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase
from cvlayer.palette import xkcd_palette
from cvlayer.typing import Color


class CvmContours(_LayerManagerMixinBase):
    def cvm_find_contours(
        self,
        name: str,
        mode=FindContoursMode.TREE,
        method=FindContoursMethod.SIMPLE,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
    ):
        with self.layer(name) as layer:
            if canvas is None:
                height, width = layer.prev_frame.shape
                canvas = zeros((height, width, 3), dtype=uint8)
            else:
                assert canvas is not None
                if len(canvas.shape) == 2:
                    canvas = cvt_color_GRAY2BGR(canvas)

            if palette is None:
                palette = xkcd_palette()

            assert isinstance(canvas, ndarray)
            assert isinstance(palette, dict)

            md = layer.param("mode").build_enumeration(mode).value
            mt = layer.param("method").build_enumeration(method).value
            amin = layer.param("amin").build_floating(area_min, 0.0, step=100.0).value
            amax = layer.param("amax").build_floating(area_max, 0.0, step=100.0).value
            contours = list(find_contours(layer.prev_frame, md, mt)[0])
            filtered_contours = list()
            for contour, color in zip(contours, palette.values()):
                area = contour_area(contour)
                if area < amin or amax < area:
                    continue

                m = moments(contour)
                if m.m00 == 0:
                    continue

                center = m.center
                draw_contour(canvas, contour, color)
                draw_point(canvas, center, color=color)
                draw_outline_text(canvas, f"{area:.2f}", center, outline_color=color)
                filtered_contours.append(contour)

            layer.frame = canvas
            layer.data = filtered_contours
        return canvas, filtered_contours

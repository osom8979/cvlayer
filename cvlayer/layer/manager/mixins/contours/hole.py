# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.contour.analysis import contour_area
from cvlayer.cv.contour.find import find_contours
from cvlayer.cv.drawable.contours import DRAW_ALL_CONTOURS, draw_contours
from cvlayer.cv.types.chain_approx import CHAIN_APPROX_SIMPLE
from cvlayer.cv.types.line_type import LINE_8
from cvlayer.cv.types.retrieval import RETR_EXTERNAL
from cvlayer.cv.types.thickness import FILLED
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmContoursHole(LayerManagerMixinBase):
    def cvm_remove_hole(
        self,
        name: str,
        area_min=0.0,
        mask_value=0,
        step=100.0,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            assert len(src.shape) == 2
            assert src.shape[0] >= 1 and src.shape[1] >= 1

            amin = layer.param("amin").build_float(area_min, 0.0, step=step).value
            color = layer.param("mask_value").build_int(mask_value, 0, 255).value

            punctures = list()
            for contour in find_contours(src, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)[0]:
                area = contour_area(contour)
                if area <= amin:
                    punctures.append(contour)

            layer.frame = draw_contours(
                src.copy(),
                punctures,
                DRAW_ALL_CONTOURS,
                color=color,
                thickness=FILLED,
                line=LINE_8,
            )

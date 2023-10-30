# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.contours import (
    FindContoursMethod,
    FindContoursMode,
    contour_area,
    find_contours,
)
from cvlayer.cv.drawable.contours import DRAW_ALL_CONTOURS, draw_contours
from cvlayer.cv.types.line_type import LINE_8
from cvlayer.cv.types.thickness import FILLED
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmPunctures(LayerManagerMixinBase):
    def cvm_remove_punctures(
        self,
        name: str,
        area_min=0.0,
        mask_val=0,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            assert len(src.shape) == 2
            assert src.shape[0] >= 1 and src.shape[1] >= 1

            amin = layer.param("amin").build_float(area_min, 0.0, step=100.0).value
            md = layer.param("mode").build_enum(FindContoursMode.EXTERNAL).value
            mt = layer.param("method").build_enum(FindContoursMethod.SIMPLE).value

            punctures = list()
            for contour in find_contours(src, md, mt)[0]:
                area = contour_area(contour)
                if area <= amin:
                    punctures.append(contour)

            layer.frame = draw_contours(
                src.copy(),
                punctures,
                DRAW_ALL_CONTOURS,
                color=mask_val,
                thickness=FILLED,
                line=LINE_8,
            )

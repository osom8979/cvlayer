# -*- coding: utf-8 -*-

from typing import Optional

from numpy import int32, uint8, zeros
from numpy.typing import NDArray

from cvlayer.cv.color import PIXEL_8BIT_MAX
from cvlayer.cv.contour.find import contour_area, find_contours
from cvlayer.cv.drawable.contours import draw_contour
from cvlayer.cv.types.chain_approx import (
    DEFAULT_CHAIN_APPROX,
    ChainApproximation,
    normalize_chain_approx,
)
from cvlayer.cv.types.line_type import LINE_8
from cvlayer.cv.types.retrieval import DEFAULT_RETRIEVAL, Retrieval, normalize_retrieval
from cvlayer.cv.types.thickness import FILLED
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmContoursExFindFilterLargest(LayerManagerMixinBase):
    def cvm_find_contours_filter_area_largest(
        self,
        name: str,
        area_min=0.0,
        area_max=0.0,
        step=10.0,
        oriented=False,
        mode=DEFAULT_RETRIEVAL,
        method=DEFAULT_CHAIN_APPROX,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame

            init_mode = Retrieval(normalize_retrieval(mode))
            init_method = ChainApproximation(normalize_chain_approx(method))

            amin = layer.param("amin").build_float(area_min, 0.0, step=step).value
            amax = layer.param("amax").build_float(area_max, 0.0, step=step).value
            retr = layer.param("mode").build_enum(init_mode).value
            approx = layer.param("method").build_enum(init_method).value

            result = find_contours(src, retr, approx)
            contours, hierarchy = result

            mask = zeros(src.shape[0:2], dtype=uint8)
            largest_contour: Optional[NDArray[int32]] = None
            largest_contour_area = 0.0

            if len(contours) >= 1:
                areas = map(lambda c: contour_area(c, oriented), contours)

                for contour, area in zip(contours, areas):
                    if amin != 0.0 and area < amin:
                        continue
                    if amax != 0.0 and amax < area:
                        continue

                    if largest_contour is None or largest_contour_area < area:
                        largest_contour = contour
                        largest_contour_area = area

                if largest_contour is not None:
                    draw_contour(
                        image=mask,
                        contour=largest_contour,
                        color=PIXEL_8BIT_MAX,
                        thickness=FILLED,
                        line=LINE_8,
                    )

            layer.frame = mask
            layer.data = largest_contour

        return mask, largest_contour

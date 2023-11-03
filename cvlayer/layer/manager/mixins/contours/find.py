# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.contour.find import find_contours
from cvlayer.cv.drawable.contours import DRAW_ALL_CONTOURS, draw_contours
from cvlayer.cv.drawable.defaults import (
    DEFAULT_COLOR,
    DEFAULT_LINE_TYPE,
    DEFAULT_THICKNESS,
)
from cvlayer.cv.types.chain_approx import (
    DEFAULT_CHAIN_APPROX,
    ChainApproximation,
    normalize_chain_approx,
)
from cvlayer.cv.types.retrieval import DEFAULT_RETRIEVAL, Retrieval, normalize_retrieval
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase
from cvlayer.math.climit import INT_MAX
from cvlayer.typing import PointI


class CvmContoursFind(LayerManagerMixinBase):
    def cvm_find_contours(
        self,
        name: str,
        mode=DEFAULT_RETRIEVAL,
        method=DEFAULT_CHAIN_APPROX,
        canvas: Optional[NDArray] = None,
        index=DRAW_ALL_CONTOURS,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        max_level=INT_MAX,
        offset: Optional[PointI] = None,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame

            init_mode = Retrieval(normalize_retrieval(mode))
            init_method = ChainApproximation(normalize_chain_approx(method))

            retr = layer.param("mode").build_enum(init_mode).value
            approx = layer.param("method").build_enum(init_method).value
            idx = layer.param("index").build_int(index, DRAW_ALL_CONTOURS).value

            result = find_contours(src, retr, approx)
            contours, hierarchy = result

            if canvas is None:
                canvas = src.copy()

            layer.frame = draw_contours(
                canvas,
                contours,
                idx,
                color,
                thickness,
                line,
                hierarchy,
                max_level,
                offset,
            )
            layer.data = contours

        return canvas, contours

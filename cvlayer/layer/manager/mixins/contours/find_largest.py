# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.color import PIXEL_8BIT_MAX
from cvlayer.cv.contour.find_largest import (
    DISABLE_AREA_FILTER,
    find_contours_filter_area_largest,
)
from cvlayer.cv.types.chain_approx import (
    DEFAULT_CHAIN_APPROX,
    ChainApproximation,
    normalize_chain_approx,
)
from cvlayer.cv.types.retrieval import DEFAULT_RETRIEVAL, Retrieval, normalize_retrieval
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmContoursFindLargest(LayerManagerMixinBase):
    def cvm_find_contours_filter_area_largest(
        self,
        name: str,
        area_min=DISABLE_AREA_FILTER,
        area_max=DISABLE_AREA_FILTER,
        step=10.0,
        oriented=False,
        mode=DEFAULT_RETRIEVAL,
        method=DEFAULT_CHAIN_APPROX,
        mask_value=PIXEL_8BIT_MAX,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame

            init_mode = Retrieval(normalize_retrieval(mode))
            init_method = ChainApproximation(normalize_chain_approx(method))

            amin = layer.param("area_min").build_float(area_min, -1.0, step=step).value
            amax = layer.param("area_max").build_float(area_max, -1.0, step=step).value
            retr = layer.param("mode").build_enum(init_mode).value
            approx = layer.param("method").build_enum(init_method).value
            o = layer.param("oriented").build_bool(oriented).value
            mval = layer.param("mask_value").build_uint(mask_value).value

            result = find_contours_filter_area_largest(
                image=src,
                mode=retr,
                method=approx,
                area_oriented=o,
                area_min=amin,
                area_max=amax,
                mask_value=mval,
            )
            mask, contour, area = result

            layer.frame = mask
            layer.data = contour

        return result

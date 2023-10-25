# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.pyramid import pyr_mean_shift_filtering
from cvlayer.cv.term_criteria import TermCriteria, TermCriteriaType
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmPyramid(LayerManagerMixinBase):
    def cvm_pyr_mean_shift_filtering(
        self,
        name: str,
        sp=20.0,
        sr=20.0,
        max_level=1,
        criteria_type=TermCriteriaType.COUNT_EPS,
        max_count=5,
        epsilon=1.0,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            s = layer.param("sp").build_float(sp, 0.0, 100.0, step=1.0).value
            c = layer.param("sr").build_float(sr, 0.0, 100.0, step=1.0).value
            ml = layer.param("max_level").build_uint(max_level, 1, 10).value
            tt = layer.param("term_type").build_enum(criteria_type).value
            tmc = layer.param("term_max_count").build_uint(max_count, 1).value
            te = layer.param("term_epsilon").build_float(epsilon, 1.0).value
            tc = TermCriteria(tt, tmc, te)
            src = frame if frame is not None else layer.prev_frame
            result = pyr_mean_shift_filtering(src, s, c, ml, tc)
            layer.frame = result
        return result

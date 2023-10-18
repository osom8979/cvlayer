# -*- coding: utf-8 -*-

from cvlayer.cv.pyramid import (
    DEFAULT_COLOR_WINDOW_RADIUS,
    DEFAULT_MAX_LEVEL,
    DEFAULT_SPATIAL_WINDOW_RADIUS,
    DEFAULT_TERM_CRITERIA_EPSILON,
    DEFAULT_TERM_CRITERIA_MAX_COUNT,
    DEFAULT_TERM_CRITERIA_TYPE,
    pyr_mean_shift_filtering,
)
from cvlayer.cv.term_criteria import TermCriteria
from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase


class CvmPyramid(_LayerManagerMixinBase):
    def cvm_pyr_mean_shift_filtering(
        self,
        sp=DEFAULT_SPATIAL_WINDOW_RADIUS,
        sr=DEFAULT_COLOR_WINDOW_RADIUS,
        max_level=DEFAULT_MAX_LEVEL,
        criteria_type=DEFAULT_TERM_CRITERIA_TYPE,
        max_count=DEFAULT_TERM_CRITERIA_MAX_COUNT,
        epsilon=DEFAULT_TERM_CRITERIA_EPSILON,
    ):
        with self.layer("...") as layer:
            s = layer.param("sp").build_floating(sp, 0.0, 100.0, step=1.0).value
            c = layer.param("sr").build_floating(sr, 0.0, 100.0, step=1.0).value
            ml = layer.param("max").build_unsigned(max_level, 1, 10).value
            ct = layer.param("type").build_enumeration(criteria_type).value
            mc = layer.param("max_count").build_unsigned(max_count, 1).value
            e = layer.param("epsilon").build_floating(epsilon, 1.0).value
            tc = TermCriteria(ct, mc, e)
            result = pyr_mean_shift_filtering(layer.prev_frame, s, c, ml, tc)
            layer.frame = result
        return result

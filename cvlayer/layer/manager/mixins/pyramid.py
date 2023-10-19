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
        name: str,
        sp=DEFAULT_SPATIAL_WINDOW_RADIUS,
        sr=DEFAULT_COLOR_WINDOW_RADIUS,
        max_level=DEFAULT_MAX_LEVEL,
        criteria_type=DEFAULT_TERM_CRITERIA_TYPE,
        max_count=DEFAULT_TERM_CRITERIA_MAX_COUNT,
        epsilon=DEFAULT_TERM_CRITERIA_EPSILON,
    ):
        with self.layer(name) as layer:
            spatial = layer.param("sp").build_floating(sp, 0.0, 100.0, step=1.0).value
            color = layer.param("sr").build_floating(sr, 0.0, 100.0, step=1.0).value
            ml = layer.param("max_level").build_unsigned(max_level, 1, 10).value
            tt = layer.param("term_type").build_enumeration(criteria_type).value
            tmc = layer.param("term_max_count").build_unsigned(max_count, 1).value
            te = layer.param("term_epsilon").build_floating(epsilon, 1.0).value
            tc = TermCriteria(tt, tmc, te)
            result = pyr_mean_shift_filtering(layer.prev_frame, spatial, color, ml, tc)
            layer.frame = result
        return result

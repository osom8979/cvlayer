# -*- coding: utf-8 -*-

from cvlayer.cv.kmeans import (
    DEFAULT_ATTEMPTS,
    DEFAULT_TERM_CRITERIA_EPSILON,
    DEFAULT_TERM_CRITERIA_MAX_COUNT,
    DEFAULT_TERM_CRITERIA_TYPE,
    KmeansFlags,
    color_quantization,
)
from cvlayer.cv.term_criteria import TermCriteria
from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase


class CvmKmeans(_LayerManagerMixinBase):
    def cvm_color_quantization(
        self,
        name: str,
        k=3,
        criteria_type=DEFAULT_TERM_CRITERIA_TYPE,
        max_count=DEFAULT_TERM_CRITERIA_MAX_COUNT,
        epsilon=DEFAULT_TERM_CRITERIA_EPSILON,
        attempts=DEFAULT_ATTEMPTS,
        flags=KmeansFlags.PP_CENTERS,
    ):
        with self.layer(name) as layer:
            k = layer.param("k").build_unsigned(k, 1).value
            tt = layer.param("term_type").build_enumeration(criteria_type).value
            tmc = layer.param("term_max_count").build_unsigned(max_count, 1).value
            te = layer.param("term_epsilon").build_floating(epsilon, 1.0).value
            a = layer.param("attempts").build_unsigned(attempts, 1).value
            f = layer.param("flags").build_enumeration(flags).value
            tc = TermCriteria(tt, tmc, te)
            result = color_quantization(layer.prev_frame, k, None, tc, a, f)
            layer.frame = result
        return result

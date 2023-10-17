# -*- coding: utf-8 -*-

from cvlayer.cv.kmeans import (
    DEFAULT_ATTEMPTS,
    DEFAULT_CRITERIA_EPSILON,
    DEFAULT_CRITERIA_MAX_COUNT,
    DEFAULT_CRITERIA_TYPE,
    KmeansFlags,
    TermCriteria,
    color_quantization,
)
from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase


class CvmKmeans(_LayerManagerMixinBase):
    def cvm_color_quantization(
        self,
        k=3,
        criteria_type=DEFAULT_CRITERIA_TYPE,
        max_count=DEFAULT_CRITERIA_MAX_COUNT,
        epsilon=DEFAULT_CRITERIA_EPSILON,
        attempts=DEFAULT_ATTEMPTS,
        flags=KmeansFlags.PP_CENTERS,
    ):
        with self.layer("cvm_color_quantization") as layer:
            k = layer.param("k").build_unsigned(k, 1).value
            ct = layer.param("type").build_enumeration(criteria_type).value
            mc = layer.param("max_count").build_unsigned(max_count, 1).value
            e = layer.param("epsilon").build_floating(epsilon, 1.0).value
            a = layer.param("attempts").build_unsigned(attempts, 1).value
            f = layer.param("flags").build_enumeration(flags).value
            tc = TermCriteria(ct, mc, e)
            result = color_quantization(layer.prev_frame, k, None, tc, a, f)
            layer.frame = result
        return result

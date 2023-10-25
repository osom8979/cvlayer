# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.kmeans import KmeansFlags, color_quantization
from cvlayer.cv.term_criteria import TermCriteria, TermCriteriaType
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmKmeans(LayerManagerMixinBase):
    def cvm_color_quantization(
        self,
        name: str,
        k=3,
        criteria_type=TermCriteriaType.COUNT_EPS,
        max_count=10,
        epsilon=1.0,
        attempts=10,
        flags=KmeansFlags.PP_CENTERS,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            k = layer.param("k").build_uint(k, 1).value
            tt = layer.param("term_type").build_enum(criteria_type).value
            tmc = layer.param("term_max_count").build_uint(max_count, 1).value
            te = layer.param("term_epsilon").build_float(epsilon, 1.0).value
            a = layer.param("attempts").build_uint(attempts, 1).value
            f = layer.param("flags").build_enum(flags).value
            tc = TermCriteria(tt, tmc, te)
            src = frame if frame is not None else layer.prev_frame
            result = color_quantization(src, k, None, tc, a, f)
            layer.frame = result
        return result

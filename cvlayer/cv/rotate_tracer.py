# -*- coding: utf-8 -*-

from cvlayer.rotate.rotate_tracer import (
    DEFAULT_ABNORMAL_DEGREES_DELTA,
    DEFAULT_MAX_ABNORMAL_COUNT,
    DEFAULT_MAX_MISSING_COUNT,
    DEFAULT_MAX_STABLE_COUNT,
    DEFAULT_STABLE_DEGREES_DELTA,
    RotateTracer,
)


class CvlRotateTracer:
    @staticmethod
    def cvl_create_rotate_tracer(
        max_missing_count=DEFAULT_MAX_MISSING_COUNT,
        max_stable_count=DEFAULT_MAX_STABLE_COUNT,
        stable_degrees_delta=DEFAULT_STABLE_DEGREES_DELTA,
        max_abnormal_count=DEFAULT_MAX_ABNORMAL_COUNT,
        abnormal_degrees_delta=DEFAULT_ABNORMAL_DEGREES_DELTA,
    ):
        return RotateTracer(
            max_missing_count=max_missing_count,
            max_stable_count=max_stable_count,
            stable_degrees_delta=stable_degrees_delta,
            max_abnormal_count=max_abnormal_count,
            abnormal_degrees_delta=abnormal_degrees_delta,
        )

# -*- coding: utf-8 -*-

from cvlayer.cv.rotate_tracer import DEFAULT_MAX_MISSING_COUNT, RotateTracer


class CvlRotateTracer:
    @staticmethod
    def cvl_create_rotate_tracer(max_missing_count=DEFAULT_MAX_MISSING_COUNT):
        return RotateTracer(max_missing_count)

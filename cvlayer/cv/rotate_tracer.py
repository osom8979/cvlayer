# -*- coding: utf-8 -*-

from typing import Optional

from cvlayer.rotate.rotate_tracer import RotateTracer


class CvlRotateTracer:
    @staticmethod
    def cvl_create_rotate_tracer(history: Optional[int] = None):
        return RotateTracer(history)

# -*- coding: utf-8 -*-

from cvlayer.math.norm import l1_norm, l2_norm, max_norm
from cvlayer.typing import NumberT


class CvlNorm:
    @staticmethod
    def cvl_l1_norm(x1: NumberT, y1: NumberT, x2: NumberT, y2: NumberT):
        return l1_norm(x1, y1, x2, y2)

    @staticmethod
    def cvl_l2_norm(x1: NumberT, y1: NumberT, x2: NumberT, y2: NumberT):
        return l2_norm(x1, y1, x2, y2)

    @staticmethod
    def cvl_max_norm(x1: NumberT, y1: NumberT, x2: NumberT, y2: NumberT):
        return max_norm(x1, y1, x2, y2)

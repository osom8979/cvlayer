# -*- coding: utf-8 -*-

from enum import Enum

import cv2

from cvlayer.math.norm import l1_norm, l2_norm, max_norm
from cvlayer.typing import NumberT

assert cv2.NORM_HAMMING2 == cv2.NORM_TYPE_MASK


class NormType(Enum):
    INF = cv2.NORM_INF
    L1 = cv2.NORM_L1
    L2 = cv2.NORM_L2
    L2SQR = cv2.NORM_L2SQR
    HAMMING = cv2.NORM_HAMMING
    HAMMING2 = cv2.NORM_HAMMING2

    TYPE_MASK = cv2.NORM_TYPE_MASK
    # bit-mask which can be used to separate norm type from norm flags

    RELATIVE = cv2.NORM_RELATIVE
    MINMAX = cv2.NORM_MINMAX


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

# -*- coding: utf-8 -*-

from cvlayer.geometry.iou import calculate_iou
from cvlayer.typing import RectT


class CvlIou:
    @staticmethod
    def cvl_calculate_iou(rect1: RectT, rect2: RectT):
        return calculate_iou(rect1, rect2)

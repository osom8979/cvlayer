# -*- coding: utf-8 -*-

from math import sqrt

from cvlayer.typing import NumberT


def l1_norm(x1: NumberT, y1: NumberT, x2: NumberT, y2: NumberT) -> NumberT:
    return abs(x2 - x1) + abs(y2 - y1)


def l2_norm(x1: NumberT, y1: NumberT, x2: NumberT, y2: NumberT) -> float:
    return sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))


def max_norm(x1: NumberT, y1: NumberT, x2: NumberT, y2: NumberT) -> NumberT:
    return max(abs(x2 - x1), abs(y2 - y1))

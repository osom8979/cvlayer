# -*- coding: utf-8 -*-

from math import tan
from typing import Tuple


def calculate_line_with_point_and_angle(
    pivot_x: float,
    pivot_y: float,
    radian: float,
    x1: float,
    x2: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    * x1
     \
      \
       * pivot(x, y)
        \
         \
          * x2
    """
    a = tan(radian)
    b = pivot_y - a * pivot_x
    # y = ax + b
    y1 = a * x1 + b
    y2 = a * x2 + b
    return (x1, y1), (x2, y2)


class GeneralForm:
    """
    Ax + By + C = 0
    """

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    @classmethod
    def from_points(cls, x1, y1, x2, y2):
        """
        (y1 - y2)x + (x2 - x1)y + (x1y2 - x2y1) = 0
        """
        a = y1 - y2
        b = x2 - x1
        c = (x1 * y2) - (x2 * y1)
        return cls(a, b, c)

    def calc_x(self, y):
        return -1 * ((self.b * y) + self.c) / self.a

    def calc_y(self, x):
        return -1 * ((self.a * x) + self.c) / self.b

    @property
    def slope(self):
        """
        y = - (Ax - C) / B
        """
        return -1 * (self.a / self.b)

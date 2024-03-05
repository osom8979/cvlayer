# -*- coding: utf-8 -*-

from math import cos, radians, sin
from typing import Tuple


def polar_to_cartesian(
    center: Tuple[float, float],
    distance: float,
    degrees: float,
) -> Tuple[float, float]:
    x = center[0] + distance * cos(radians(degrees))
    y = center[1] + distance * sin(radians(degrees))
    return x, y

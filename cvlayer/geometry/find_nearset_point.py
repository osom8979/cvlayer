# -*- coding: utf-8 -*-

from cvlayer.math.norm import l2_norm
from cvlayer.types import Point


def find_nearest_point_index(pivot: Point, *points: Point) -> int:
    norms = [l2_norm(pivot[0], pivot[1], p[0], p[1]) for p in points]
    return norms.index(min(norms))


def find_nearest_point(pivot: Point, *points: Point) -> Point:
    return points[find_nearest_point_index(pivot, *points)]

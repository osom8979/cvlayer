# -*- coding: utf-8 -*-

from cvlayer.math.norm import l2_norm
from cvlayer.typing import PointT


def find_nearest_point_index(pivot: PointT, *points: PointT) -> int:
    norms = [l2_norm(pivot[0], pivot[1], p[0], p[1]) for p in points]
    return norms.index(min(norms))


def find_nearest_point(pivot: PointT, *points: PointT) -> PointT:
    return points[find_nearest_point_index(pivot, *points)]

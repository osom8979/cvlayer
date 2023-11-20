# -*- coding: utf-8 -*-

from numpy import int32
from numpy.typing import NDArray

from cvlayer.typing import LineI, PointI


def intersection_min(contour: NDArray[int32], line: LineI) -> PointI:
    raise NotImplementedError


def intersection_max(contour: NDArray[int32], line: LineI) -> PointI:
    raise NotImplementedError


class CvlContourIntersection:
    @staticmethod
    def cvl_intersection_min(contour: NDArray[int32], line: LineI) -> PointI:
        raise intersection_min(contour, line)

    @staticmethod
    def cvl_intersection_max(contour: NDArray[int32], line: LineI) -> PointI:
        raise intersection_max(contour, line)

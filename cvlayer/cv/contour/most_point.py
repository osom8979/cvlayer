# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.typing import PointI


def find_leftmost_point(contour: NDArray) -> PointI:
    p = contour[contour[:, :, 0].argmin()][0]
    return int(p[0]), int(p[1])


def find_rightmost_point(contour: NDArray) -> PointI:
    p = contour[contour[:, :, 0].argmax()][0]
    return int(p[0]), int(p[1])


def find_topmost_point(contour: NDArray) -> PointI:
    p = contour[contour[:, :, 1].argmin()][0]
    return int(p[0]), int(p[1])


def find_bottommost_point(contour: NDArray) -> PointI:
    p = contour[contour[:, :, 1].argmax()][0]
    return int(p[0]), int(p[1])


class CvlContourMostPoint:
    @staticmethod
    def cvl_find_leftmost_point(contour: NDArray):
        return find_leftmost_point(contour)

    @staticmethod
    def cvl_find_rightmost_point(contour: NDArray):
        return find_rightmost_point(contour)

    @staticmethod
    def cvl_find_topmost_point(contour: NDArray):
        return find_topmost_point(contour)

    @staticmethod
    def cvl_find_bottommost_point(contour: NDArray):
        return find_bottommost_point(contour)

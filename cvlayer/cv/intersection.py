# -*- coding: utf-8 -*-

from typing import Optional

from numpy import argmax, argmin, int64, nonzero, uint8, zeros_like
from numpy.typing import NDArray

from cvlayer.cv.bitwise import bitwise_and
from cvlayer.cv.drawable.line import draw_line
from cvlayer.cv.types.line_type import LINE_4
from cvlayer.typing import LineI, PointI


class IntersectionResult:
    _leftmost: Optional[PointI]
    _rightmost: Optional[PointI]
    _topmost: Optional[PointI]
    _bottommost: Optional[PointI]

    def __init__(self, yis: NDArray[int64], xis: NDArray[int64]):
        self.yis = yis
        self.xis = xis
        self._leftmost = None
        self._rightmost = None
        self._topmost = None
        self._bottommost = None

    @property
    def leftmost(self) -> PointI:
        if self._leftmost is None:
            index = argmin(self.xis)
            self._leftmost = int(self.xis[index]), int(self.yis[index])
        assert self._leftmost is not None
        return self._leftmost

    @property
    def rightmost(self) -> PointI:
        if self._rightmost is None:
            index = argmax(self.xis)
            self._rightmost = int(self.xis[index]), int(self.yis[index])
        assert self._rightmost is not None
        return self._rightmost

    @property
    def topmost(self) -> PointI:
        if self._topmost is None:
            index = argmin(self.yis)
            self._topmost = int(self.xis[index]), int(self.yis[index])
        assert self._topmost is not None
        return self._topmost

    @property
    def bottommost(self) -> PointI:
        if self._bottommost is None:
            index = argmax(self.yis)
            self._bottommost = int(self.xis[index]), int(self.yis[index])
        assert self._bottommost is not None
        return self._bottommost


def intersection_mask_line(
    mask: NDArray[uint8],
    line: LineI,
    line_type=LINE_4,
) -> IntersectionResult:
    assert len(mask.shape) == 2
    canvas = zeros_like(mask, dtype=uint8)
    draw_line(canvas, line[0], line[1], 255, 1, line=line_type)
    result = bitwise_and(canvas, mask)
    yis, xis = nonzero(result)
    if yis.size == 0:
        raise IndexError("Not found intersection")
    assert yis.size == xis.size
    return IntersectionResult(yis, xis)


class CvlIntersection:
    @staticmethod
    def cvl_intersection_mask_line(mask: NDArray[uint8], line: LineI, line_type=LINE_4):
        return intersection_mask_line(mask, line, line_type)

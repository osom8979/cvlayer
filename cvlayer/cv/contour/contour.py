# -*- coding: utf-8 -*-

from typing import Optional

from numpy import int32
from numpy.typing import NDArray

from cvlayer.cv.contour.edge import (
    find_bottommost_point,
    find_leftmost_point,
    find_rightmost_point,
    find_topmost_point,
)
from cvlayer.cv.contour.find import contour_area
from cvlayer.cv.contour.moments import Moments, moments
from cvlayer.typing import PointI


class Contour:
    _array: NDArray[int32]

    _m: Optional[Moments]
    _area: Optional[float]

    _leftmost_point: Optional[PointI]
    _rightmost_point: Optional[PointI]
    _topmost_point: Optional[PointI]
    _bottommost_point: Optional[PointI]

    def __init__(
        self,
        array: NDArray[int32],
        copy=True,
        write=False,
        binary_image=False,
        oriented=False,
    ):
        self._array = array.copy() if copy else array
        self._array.setflags(write=write)

        self._m = None
        self._binary_image = binary_image

        self._area = None
        self._oriented = oriented

        self._leftmost_point = None
        self._rightmost_point = None
        self._topmost_point = None
        self._bottommost_point = None

    @property
    def array(self) -> NDArray[int32]:
        return self._array

    @property
    def moments(self) -> Moments:
        if self._m is None:
            self._m = moments(self._array, self._binary_image)
        assert self._m is not None
        return self._m

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = contour_area(self._array, self._oriented)
        assert self._area is not None
        return self._area

    @property
    def leftmost_point(self) -> PointI:
        if self._leftmost_point is None:
            self._leftmost_point = find_leftmost_point(self._array)
        assert self._leftmost_point is not None
        return self._leftmost_point

    @property
    def rightmost_point(self) -> PointI:
        if self._rightmost_point is None:
            self._rightmost_point = find_rightmost_point(self._array)
        assert self._rightmost_point is not None
        return self._rightmost_point

    @property
    def topmost_point(self) -> PointI:
        if self._topmost_point is None:
            self._topmost_point = find_topmost_point(self._array)
        assert self._topmost_point is not None
        return self._topmost_point

    @property
    def bottommost_point(self) -> PointI:
        if self._bottommost_point is None:
            self._bottommost_point = find_bottommost_point(self._array)
        assert self._bottommost_point is not None
        return self._bottommost_point


class CvlContourContour:
    @staticmethod
    def cvl_create_contour(
        array: NDArray[int32],
        copy=True,
        write=False,
        binary_image=False,
    ):
        return Contour(array, copy, write, binary_image)

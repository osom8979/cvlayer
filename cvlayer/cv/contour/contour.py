# -*- coding: utf-8 -*-

from typing import Optional

from numpy import int32
from numpy.typing import NDArray

from cvlayer.cv.contour.analysis import contour_area
from cvlayer.cv.contour.moments import Moments, moments
from cvlayer.cv.contour.most_point import (
    find_bottommost_point,
    find_leftmost_point,
    find_rightmost_point,
    find_topmost_point,
)
from cvlayer.typing import PointI, PolygonI


class Contour:
    _array: NDArray[int32]

    _m: Optional[Moments]
    _area: Optional[float]

    _leftmost_point: Optional[PointI]
    _rightmost_point: Optional[PointI]
    _topmost_point: Optional[PointI]
    _bottommost_point: Optional[PointI]

    _polygon0: Optional[PolygonI]

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

        self._polygon0 = None

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

    @area.setter
    def area(self, value: Optional[float]) -> None:
        self._area = value

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

    @property
    def width(self):
        return self.rightmost_point[0] - self.leftmost_point[0]

    @property
    def height(self):
        return self.bottommost_point[1] - self.topmost_point[1]

    @property
    def polygon0(self) -> PolygonI:
        if self._polygon0 is None:
            assert len(self._array.shape) == 3
            assert self._array.shape[0] >= 4  # points
            assert self._array.shape[1] >= 1  # -
            assert self._array.shape[2] == 2  # x, y
            points = self._array[:, 0, :].tolist()
            self._polygon0 = list(map(lambda x: (x[0], x[1]), points))
        assert self._polygon0 is not None
        return self._polygon0


class CvlContourContour:
    @staticmethod
    def cvl_create_contour(
        array: NDArray[int32],
        copy=True,
        write=False,
        binary_image=False,
    ):
        return Contour(array, copy, write, binary_image)

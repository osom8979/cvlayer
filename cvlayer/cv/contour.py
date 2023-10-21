# -*- coding: utf-8 -*-

from typing import Optional

from numpy import int32
from numpy.typing import NDArray

from cvlayer.cv.contours_moments import Moments, moments


class Contour:
    _array: NDArray[int32]
    _m: Optional[Moments]

    def __init__(
        self,
        array: NDArray[int32],
        copy=True,
        write=False,
        *,
        binary_image=False,
    ):
        self._array = array.copy() if copy else array
        self._array.setflags(write=write)
        self._m = None
        self._binary_image = binary_image

    @property
    def array(self) -> NDArray[int32]:
        return self._array

    @property
    def moments(self) -> Moments:
        if self._m is None:
            self._m = moments(self._array, self._binary_image)
        assert self._m is not None
        return self._m

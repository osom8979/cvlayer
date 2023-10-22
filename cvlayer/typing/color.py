# -*- coding: utf-8 -*-

from typing import Sequence, Tuple, Union

_Blue = int
_Green = int
_Red = int

BGR = Tuple[_Blue, _Green, _Red]
RGB = Tuple[_Red, _Green, _Blue]

Color = Sequence[float]
ColorLike = Union[int, float, str, Sequence[int], Sequence[float]]

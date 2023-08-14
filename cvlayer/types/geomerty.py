# -*- coding: utf-8 -*-

from typing import List, Tuple, TypeVar, Union

Number = Union[int, float]
NumberT = TypeVar("NumberT", int, float)

Point = Tuple[
    NumberT,  # X
    NumberT,  # Y
]
PointInt = Tuple[int, int]
PointFloat = Tuple[float, float]

Size = Tuple[
    NumberT,  # Width
    NumberT,  # Height
]
SizeInt = Tuple[int, int]
SizeFloat = Tuple[float, float]

Polygon = List[Point]

Rect = Tuple[
    NumberT,  # X1 - Left
    NumberT,  # Y1 - Top
    NumberT,  # X2 - Right
    NumberT,  # Y2 - Bottom
]
RectInt = Tuple[int, int, int, int]
RectFloat = Tuple[float, float, float, float]

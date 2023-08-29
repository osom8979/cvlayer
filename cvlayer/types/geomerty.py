# -*- coding: utf-8 -*-

from typing import List, Tuple, TypeVar, Union

Number = Union[int, float]
NumberT = TypeVar("NumberT", int, float)

PointT = Tuple[NumberT, NumberT]
PointInt = Tuple[int, int]
PointFloat = Tuple[float, float]

SizeT = Tuple[NumberT, NumberT]
SizeInt = Tuple[int, int]
SizeFloat = Tuple[float, float]

LineT = Tuple[PointT, PointT]
LineInt = Tuple[PointInt, PointInt]
LineFloat = Tuple[PointFloat, PointFloat]

RectT = Tuple[NumberT, NumberT, NumberT, NumberT]
RectInt = Tuple[int, int, int, int]
RectFloat = Tuple[float, float, float, float]

PolygonT = List[PointT]
PolygonInt = List[PointInt]
PolygonFloat = List[PointFloat]

Scalar = Tuple[float, float, float, float]

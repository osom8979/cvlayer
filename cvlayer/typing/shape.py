# -*- coding: utf-8 -*-

from typing import Sequence, Tuple, TypeVar, Union

Number = Union[int, float]
NumberT = TypeVar("NumberT", int, float)

PointT = Tuple[NumberT, NumberT]
PointN = Tuple[Number, Number]
PointI = Tuple[int, int]
PointF = Tuple[float, float]

SizeT = Tuple[NumberT, NumberT]
SizeN = Tuple[Number, Number]
SizeI = Tuple[int, int]
SizeF = Tuple[float, float]

LineT = Tuple[PointT, PointT]
LineN = Tuple[PointN, PointN]
LineI = Tuple[PointI, PointI]
LineF = Tuple[PointF, PointF]

RectT = Tuple[NumberT, NumberT, NumberT, NumberT]
RectN = Tuple[Number, Number, Number, Number]
RectI = Tuple[int, int, int, int]
RectF = Tuple[float, float, float, float]

PolygonT = Sequence[PointT]
PolygonN = Sequence[PointN]
PolygonI = Sequence[PointI]
PolygonF = Sequence[PointF]

PerspectivePointsT = Tuple[PointT, PointT, PointT, PointT]
PerspectivePointsN = Tuple[PointN, PointN, PointN, PointN]
PerspectivePointsI = Tuple[PointI, PointI, PointI, PointI]
PerspectivePointsF = Tuple[PointF, PointF, PointF, PointF]

ScalarT = Tuple[NumberT, NumberT, NumberT, NumberT]
ScalarN = Tuple[Number, Number, Number, Number]
ScalarI = Tuple[int, int, int, int]
ScalarF = Tuple[float, float, float, float]

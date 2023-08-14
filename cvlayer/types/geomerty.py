# -*- coding: utf-8 -*-

from typing import List, Tuple, TypeVar, Union

Number = Union[int, float]
NumberT = TypeVar("NumberT", int, float)

Point = Tuple[NumberT, NumberT]
Polygon = List[Point]

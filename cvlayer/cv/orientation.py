# -*- coding: utf-8 -*-

from enum import Enum, auto, unique


@unique
class Orientation(Enum):
    Horizontal = auto()
    Vertical = auto()

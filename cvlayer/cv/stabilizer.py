# -*- coding: utf-8 -*-

from enum import Enum, auto, unique
from typing import Final


@unique
class StabilizerBorderTypes(Enum):
    Black = auto()
    Reflect = auto()
    Reflect_101 = auto()
    Replicate = auto()
    Wrap = auto()


DEFAULT_SMOOTHING_RADIUS: Final[int] = 25
DEFAULT_BORDER_TYPE: Final[StabilizerBorderTypes] = StabilizerBorderTypes.Black
DEFAULT_BORDER_SIZE: Final[int] = 0
DEFAULT_CROP_N_ZOOM: Final[bool] = False
DEFAULT_LOGGING: Final[bool] = False

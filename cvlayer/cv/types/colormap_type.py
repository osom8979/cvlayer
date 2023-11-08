# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Union

import cv2

from cvlayer.enum.utils import make_name_map

COLORMAP_AUTUMN: Final[int] = cv2.COLORMAP_AUTUMN
COLORMAP_BONE: Final[int] = cv2.COLORMAP_BONE
COLORMAP_JET: Final[int] = cv2.COLORMAP_JET
COLORMAP_WINTER: Final[int] = cv2.COLORMAP_WINTER
COLORMAP_RAINBOW: Final[int] = cv2.COLORMAP_RAINBOW
COLORMAP_OCEAN: Final[int] = cv2.COLORMAP_OCEAN
COLORMAP_SUMMER: Final[int] = cv2.COLORMAP_SUMMER
COLORMAP_SPRING: Final[int] = cv2.COLORMAP_SPRING
COLORMAP_COOL: Final[int] = cv2.COLORMAP_COOL
COLORMAP_HSV: Final[int] = cv2.COLORMAP_HSV
COLORMAP_PINK: Final[int] = cv2.COLORMAP_PINK
COLORMAP_HOT: Final[int] = cv2.COLORMAP_HOT
COLORMAP_PARULA: Final[int] = cv2.COLORMAP_PARULA
COLORMAP_MAGMA: Final[int] = cv2.COLORMAP_MAGMA
COLORMAP_INFERNO: Final[int] = cv2.COLORMAP_INFERNO
COLORMAP_PLASMA: Final[int] = cv2.COLORMAP_PLASMA
COLORMAP_VIRIDIS: Final[int] = cv2.COLORMAP_VIRIDIS
COLORMAP_CIVIDIS: Final[int] = cv2.COLORMAP_CIVIDIS
COLORMAP_TWILIGHT: Final[int] = cv2.COLORMAP_TWILIGHT
COLORMAP_TWILIGHT_SHIFTED: Final[int] = cv2.COLORMAP_TWILIGHT_SHIFTED
COLORMAP_TURBO: Final[int] = cv2.COLORMAP_TURBO
COLORMAP_DEEPGREEN: Final[int] = cv2.COLORMAP_DEEPGREEN


@unique
class ColormapType(Enum):
    AUTUMN = COLORMAP_AUTUMN
    BONE = COLORMAP_BONE
    JET = COLORMAP_JET
    WINTER = COLORMAP_WINTER
    RAINBOW = COLORMAP_RAINBOW
    OCEAN = COLORMAP_OCEAN
    SUMMER = COLORMAP_SUMMER
    SPRING = COLORMAP_SPRING
    COOL = COLORMAP_COOL
    HSV = COLORMAP_HSV
    PINK = COLORMAP_PINK
    HOT = COLORMAP_HOT
    PARULA = COLORMAP_PARULA
    MAGMA = COLORMAP_MAGMA
    INFERNO = COLORMAP_INFERNO
    PLASMA = COLORMAP_PLASMA
    VIRIDIS = COLORMAP_VIRIDIS
    CIVIDIS = COLORMAP_CIVIDIS
    TWILIGHT = COLORMAP_TWILIGHT
    TWILIGHT_SHIFTED = COLORMAP_TWILIGHT_SHIFTED
    TURBO = COLORMAP_TURBO
    DEEPGREEN = COLORMAP_DEEPGREEN


ColormapTypeLike = Union[ColormapType, str, int]

COLORMAP_TYPE_MAP: Final[Dict[str, ColormapType]] = make_name_map(ColormapType, False)


def normalize_colormap_type(colormap: ColormapTypeLike) -> int:
    if isinstance(colormap, ColormapType):
        return colormap.value
    elif isinstance(colormap, str):
        return COLORMAP_TYPE_MAP[colormap].value
    elif isinstance(colormap, int):
        return colormap
    else:
        raise TypeError(f"Unsupported colormap type: {type(colormap).__name__}")

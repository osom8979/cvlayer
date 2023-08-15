# -*- coding: utf-8 -*-

from enum import Enum, auto, unique
from functools import lru_cache
from typing import Dict, Final

from cvlayer.cv.backend import (
    HIGHGUI_BACKEND_TYPE_LINUX,
    HighGuiBackend,
    highgui_backend_type,
)

KEYCODE_NULL: Final[int] = -1
KEYCODE_ESC: Final[int] = 27
KEYCODE_ENTER: Final[int] = 13


@unique
class HighGuiKeyCode(Enum):
    ARROW_UP = auto()
    ARROW_DOWN = auto()
    ARROW_LEFT = auto()
    ARROW_RIGHT = auto()


@lru_cache
def highgui_keys() -> Dict[HighGuiKeyCode, int]:
    backend_type = highgui_backend_type()
    if backend_type == HighGuiBackend.DARWIN:
        return {
            HighGuiKeyCode.ARROW_UP: 63232,
            HighGuiKeyCode.ARROW_DOWN: 63233,
            HighGuiKeyCode.ARROW_LEFT: 63234,
            HighGuiKeyCode.ARROW_RIGHT: 63235,
        }
    elif backend_type in HIGHGUI_BACKEND_TYPE_LINUX:
        return {
            HighGuiKeyCode.ARROW_UP: 65362,
            HighGuiKeyCode.ARROW_DOWN: 65364,
            HighGuiKeyCode.ARROW_LEFT: 65361,
            HighGuiKeyCode.ARROW_RIGHT: 65363,
        }
    else:
        return {}


def has_highgui_arrow_keys(keymap: Dict[HighGuiKeyCode, int]):
    if HighGuiKeyCode.ARROW_UP not in keymap:
        return False
    if HighGuiKeyCode.ARROW_DOWN not in keymap:
        return False
    if HighGuiKeyCode.ARROW_LEFT not in keymap:
        return False
    if HighGuiKeyCode.ARROW_RIGHT not in keymap:
        return False
    return True

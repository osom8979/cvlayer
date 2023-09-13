# -*- coding: utf-8 -*-

from enum import Enum, auto, unique
from functools import lru_cache
from typing import Dict, Final, List

from cvlayer.cv.backend import (
    HIGHGUI_BACKEND_TYPE_LINUX,
    HighGuiBackend,
    highgui_backend_type,
)
from cvlayer.keymap.create import (
    DEFAULT_CALLBACK_NAME_PREFIX,
    DEFAULT_CALLBACK_NAME_SUFFIX,
    create_callable_keymap,
)

KEYCODE_TIMEOUT: Final[int] = -1
KEYCODE_NULL: Final[int] = 0
KEYCODE_ENTER: Final[int] = 13
KEYCODE_ESC: Final[int] = 27


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


class CvlKeymap:
    @staticmethod
    def cvl_highgui_keys():
        return highgui_keys()

    @staticmethod
    def cvl_has_highgui_arrow_keys(keymap: Dict[HighGuiKeyCode, int]):
        return has_highgui_arrow_keys(keymap)

    @staticmethod
    def cvl_create_callable_keymap(
        obj: object,
        keymaps: Dict[str, List[int]],
        callback_name_prefix=DEFAULT_CALLBACK_NAME_PREFIX,
        callback_name_suffix=DEFAULT_CALLBACK_NAME_SUFFIX,
    ):
        return create_callable_keymap(
            obj,
            keymaps,
            callback_name_prefix,
            callback_name_suffix,
        )

# -*- coding: utf-8 -*-

from inspect import isroutine
from typing import Callable, Dict, Final, List

from cvlayer.inspect.member import get_public_instance_attributes

DEFAULT_CALLBACK_NAME_PREFIX: Final[str] = "on_keydown_"
DEFAULT_CALLBACK_NAME_SUFFIX: Final[str] = ""


def create_callable_keymap(
    obj: object,
    shortcut: Dict[str, List[int]],
    callback_name_prefix=DEFAULT_CALLBACK_NAME_PREFIX,
    callback_name_suffix=DEFAULT_CALLBACK_NAME_SUFFIX,
) -> Dict[int, Callable[[int], None]]:
    result: Dict[int, Callable[[int], None]] = dict()
    for attr, keycodes in get_public_instance_attributes(shortcut):
        callback_name = callback_name_prefix + attr + callback_name_suffix
        assert isinstance(keycodes, list)
        for keycode in keycodes:
            assert isinstance(keycode, int)
            if hasattr(obj, callback_name):
                cb = getattr(obj, callback_name)
                assert isroutine(cb)
                result[keycode] = cb
    return result

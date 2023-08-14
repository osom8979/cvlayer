# -*- coding: utf-8 -*-

from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import Dict, Final, Tuple

PALETTE_PACKAGE_PATH: Final[str] = "cvlayer.palette"


def _palette_filter(module: ModuleType, key: str) -> bool:
    if not key.isupper():
        return False

    value = getattr(module, key)
    if not isinstance(value, (tuple, list)):
        return False

    if len(value) not in (3, 4):
        return False

    if not isinstance(value[0], int):
        return False
    if not isinstance(value[1], int):
        return False
    if not isinstance(value[2], int):
        return False

    assert 0 <= value[0] <= 255
    assert 0 <= value[1] <= 255
    assert 0 <= value[2] <= 255
    return True


def _load_palette_from_module(module: ModuleType) -> Dict[str, Tuple[int, int, int]]:
    keys = list(filter(lambda x: _palette_filter(module, x), dir(module)))
    return {k: getattr(module, k) for k in keys}


def _load_palette_from_module_name(module_name: str):
    module = import_module(f"{PALETTE_PACKAGE_PATH}.{module_name}")
    return _load_palette_from_module(module)


@lru_cache
def basic_palette():
    return _load_palette_from_module_name("basic")


@lru_cache
def css4_palette():
    return _load_palette_from_module_name("css4")


@lru_cache
def extended_palette():
    return _load_palette_from_module_name("extended")


@lru_cache
def flat_palette():
    return _load_palette_from_module_name("flat")


@lru_cache
def tableau_palette():
    return _load_palette_from_module_name("tableau")


@lru_cache
def xkcd_palette():
    return _load_palette_from_module_name("xkcd")

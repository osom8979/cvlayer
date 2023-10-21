# -*- coding: utf-8 -*-

from enum import Enum, auto, unique
from functools import lru_cache
from re import sub as re_sub
from sys import platform

import cv2


@unique
class HighGuiBackend(Enum):
    UNKNOWN = auto()
    WINDOWS = auto()
    DARWIN = auto()
    LINUX_QT = auto()
    LINUX_GTK = auto()
    LINUX_UNKNOWN = auto()


HIGHGUI_BACKEND_TYPE_LINUX = (
    HighGuiBackend.LINUX_QT,
    HighGuiBackend.LINUX_GTK,
    HighGuiBackend.LINUX_UNKNOWN,
)


@lru_cache
def get_build_information() -> str:
    return cv2.getBuildInformation()


@lru_cache
def highgui_backend() -> str:
    infos = get_build_information()
    assert isinstance(infos, str)
    for line in infos.split("\n"):
        strip_line = line.strip()
        if not strip_line.startswith("GUI:"):
            continue
        return re_sub(r"GUI:[ \\t]*", "", strip_line).strip()
    return str()


@lru_cache
def highgui_backend_type() -> HighGuiBackend:
    backend = highgui_backend()
    if platform.startswith("win32"):
        return HighGuiBackend.WINDOWS
    elif platform.startswith("darwin"):
        return HighGuiBackend.DARWIN
    elif platform.startswith("linux"):
        if backend.startswith("QT"):
            return HighGuiBackend.LINUX_QT
        elif backend.startswith("GTK"):
            return HighGuiBackend.LINUX_GTK
        else:
            return HighGuiBackend.LINUX_UNKNOWN
    else:
        return HighGuiBackend.UNKNOWN


class CvlBackend:
    @staticmethod
    def cvl_get_build_information():
        return get_build_information()

    @staticmethod
    def cvl_highgui_backend():
        return highgui_backend()

    @staticmethod
    def cvl_highgui_backend_type():
        return highgui_backend_type()


if __name__ == "__main__":
    print(get_build_information())

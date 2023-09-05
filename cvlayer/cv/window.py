# -*- coding: utf-8 -*-

from copy import deepcopy
from enum import Enum, unique
from functools import wraps
from typing import Callable, Final

import cv2
from numpy.typing import NDArray

from cvlayer.cv.mouse import MouseEvent
from cvlayer.typing import RectInt

FUNC_SET_MOUSE_CALLBACK: Final[str] = "setMouseCallback"
FUNC_CREATE_TRACKBAR: Final[str] = "createTrackbar"

WINDOW_NORMAL: Final[int] = cv2.WINDOW_NORMAL
WINDOW_AUTOSIZE: Final[int] = cv2.WINDOW_AUTOSIZE
WINDOW_OPENGL: Final[int] = cv2.WINDOW_OPENGL
WINDOW_FULLSCREEN: Final[int] = cv2.WINDOW_FULLSCREEN
WINDOW_FREERATIO: Final[int] = cv2.WINDOW_FREERATIO
WINDOW_KEEPRATIO: Final[int] = cv2.WINDOW_KEEPRATIO
WINDOW_GUI_EXPANDED: Final[int] = cv2.WINDOW_GUI_EXPANDED
WINDOW_GUI_NORMAL: Final[int] = cv2.WINDOW_GUI_NORMAL


@unique
class WindowProperty(Enum):
    FULLSCREEN = cv2.WND_PROP_FULLSCREEN
    AUTOSIZE = cv2.WND_PROP_AUTOSIZE
    ASPECT_RATIO = cv2.WND_PROP_ASPECT_RATIO
    OPENGL = cv2.WND_PROP_OPENGL
    VISIBLE = cv2.WND_PROP_VISIBLE
    TOPMOST = cv2.WND_PROP_TOPMOST
    VSYNC = cv2.WND_PROP_VSYNC


def _has_set_mouse_callback() -> bool:
    return hasattr(cv2, FUNC_SET_MOUSE_CALLBACK)


def _set_mouse_callback(winname: str, callback: Callable, userdata=None) -> None:
    func = getattr(cv2, FUNC_SET_MOUSE_CALLBACK)
    func(winname, callback, userdata)


def _has_create_trackbar() -> bool:
    return hasattr(cv2, FUNC_CREATE_TRACKBAR)


def _create_trackbar(
    trackbarname: str,
    winname: str,
    value: int,
    count: int,
    callback: Callable,
) -> None:
    func = getattr(cv2, FUNC_CREATE_TRACKBAR)
    func(trackbarname, winname, value, count, callback)


class Window:
    def __init__(self, name: str, flags=WINDOW_AUTOSIZE):
        if not name:
            raise ValueError("A window name is required")

        self._name = name
        self._has_mouse = _has_set_mouse_callback()
        self._has_trackbar = _has_create_trackbar()

        cv2.namedWindow(name, flags)
        if self._has_mouse:
            _set_mouse_callback(name, self._mouse_callback)

    @property
    def has_mouse(self) -> str:
        return deepcopy(self._name)

    @property
    def name(self) -> str:
        return deepcopy(self._name)

    @staticmethod
    def start_window_thread() -> None:
        cv2.startWindowThread()

    @staticmethod
    def destroy_all_windows() -> None:
        cv2.destroyAllWindows()

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, userdata) -> None:
        assert userdata is None
        self.on_mouse(MouseEvent(event), x, y, flags)

    def on_mouse(self, event: MouseEvent, x: int, y: int, flags: int) -> None:
        pass

    def _trackbar_callback(self, name: str, value: int) -> None:
        self.on_trackbar(name, value)

    def on_trackbar(self, name: str, value: int) -> None:
        self.on_trackbar(name, value)

    def create_trackbar(self, trackbarname: str, value: int, count: int) -> None:
        _create_trackbar(
            trackbarname,
            self._name,
            value,
            count,
            wraps(self._trackbar_callback, trackbarname),
        )

    def get_trackbar_pos(self, trackbarname: str) -> int:
        return cv2.getTrackbarPos(trackbarname, self._name)

    def set_trackbar_max(self, trackbarname: str, maxval: int) -> None:
        cv2.setTrackbarMax(trackbarname, self._name, maxval)

    def set_trackbar_min(self, trackbarname: str, minval: int) -> None:
        cv2.setTrackbarMin(trackbarname, self._name, minval)

    def set_trackbar_pos(self, trackbarname: str, pos: int) -> None:
        cv2.setTrackbarPos(trackbarname, self._name, pos)

    def destroy(self) -> None:
        cv2.destroyWindow(self._name)

    def draw(self, image: NDArray) -> None:
        cv2.imshow(self._name, image)

    def move(self, x: int, y: int) -> None:
        cv2.moveWindow(self._name, x, y)

    def resize(self, width: int, height: int) -> None:
        cv2.resizeWindow(self._name, width, height)

    def get_property(self, prop: WindowProperty) -> float:
        return cv2.getWindowProperty(self._name, prop.value)

    def set_property(self, prop: WindowProperty, value: float) -> None:
        cv2.setWindowProperty(self._name, prop.value, value)

    @property
    def fullscreen(self) -> float:
        return self.get_property(WindowProperty.FULLSCREEN)

    @fullscreen.setter
    def fullscreen(self, value: float) -> None:
        self.set_property(WindowProperty.FULLSCREEN, value)

    @property
    def autosize(self) -> float:
        return self.get_property(WindowProperty.AUTOSIZE)

    @autosize.setter
    def autosize(self, value: float) -> None:
        self.set_property(WindowProperty.AUTOSIZE, value)

    @property
    def aspect_ratio(self) -> float:
        return self.get_property(WindowProperty.ASPECT_RATIO)

    @aspect_ratio.setter
    def aspect_ratio(self, value: float) -> None:
        self.set_property(WindowProperty.ASPECT_RATIO, value)

    @property
    def opengl(self) -> float:
        return self.get_property(WindowProperty.OPENGL)

    @opengl.setter
    def opengl(self, value: float) -> None:
        self.set_property(WindowProperty.OPENGL, value)

    @property
    def visible(self) -> float:
        return self.get_property(WindowProperty.VISIBLE)

    @visible.setter
    def visible(self, value: float) -> None:
        self.set_property(WindowProperty.VISIBLE, value)

    @property
    def topmost(self) -> float:
        return self.get_property(WindowProperty.TOPMOST)

    @topmost.setter
    def topmost(self, value: float) -> None:
        self.set_property(WindowProperty.TOPMOST, value)

    @property
    def vsync(self) -> float:
        return self.get_property(WindowProperty.VSYNC)

    @vsync.setter
    def vsync(self, value: float) -> None:
        self.set_property(WindowProperty.VSYNC, value)

    def set_title(self, title: str) -> None:
        cv2.setWindowTitle(self._name, title)

    def get_image_rect(self) -> RectInt:
        x1, y1, x2, y2 = cv2.getWindowImageRect(self._name)
        return x1, y1, x2, y2

    @staticmethod
    def poll_key() -> int:
        return cv2.pollKey()

    @staticmethod
    def wait_key(delay=0) -> int:
        return cv2.waitKey(delay)

    @staticmethod
    def wait_key_ex(delay=0) -> int:
        return cv2.waitKeyEx(delay)


class CvlWindow:
    @staticmethod
    def cvl_create_window(name: str, flags=WINDOW_AUTOSIZE) -> Window:
        return Window(name, flags)

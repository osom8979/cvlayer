# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

from numpy.typing import NDArray

from cvlayer.cv.mouse import EventFlags, MouseEvent


class VioInterface(metaclass=ABCMeta):
    @abstractmethod
    def on_create(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_destroy(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_frame(self, frame: NDArray) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def on_keydown(self, keycode: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def on_mouse(self, event: MouseEvent, x: int, y: int, flags: EventFlags) -> None:
        raise NotImplementedError

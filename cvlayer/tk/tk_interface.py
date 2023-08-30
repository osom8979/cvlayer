# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class TkAsyncInterface(metaclass=ABCMeta):
    @abstractmethod
    async def on_escape(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def on_keydown(self, code: str) -> None:
        raise NotImplementedError

# -*- coding: utf-8 -*-

from asyncio import AbstractEventLoop, get_event_loop, run_coroutine_threadsafe
from asyncio.exceptions import CancelledError
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional

from numpy import uint8, zeros
from numpy.typing import NDArray
from overrides import override

from cvlayer.tk.tk_interface import TkAsyncInterface
from cvlayer.tk.tk_window import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_TITLE,
    DEFAULT_WIDTH,
    DEFAULT_X,
    DEFAULT_Y,
    TkWindow,
)


class TkAsyncWindow(TkWindow):
    def __init__(
        self,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        x=DEFAULT_X,
        y=DEFAULT_Y,
        title=DEFAULT_TITLE,
        fps=DEFAULT_FPS,
        *,
        callback: Optional[TkAsyncInterface] = None,
        loop: Optional[AbstractEventLoop] = None,
    ):
        super().__init__(width, height, x, y, title, fps, use_infinity_after=False)
        self._callback = callback
        self._loop = loop if loop else get_event_loop()
        self._empty = zeros((width, height, 3), dtype=uint8)
        self._latest_frame = self._empty.copy()

    @override
    def on_grab(self) -> NDArray:
        return self._empty

    @override
    def on_escape(self) -> None:
        if not self._callback:
            return
        run_coroutine_threadsafe(self._callback.on_escape(), self._loop)

    @override
    def on_keydown(self, code: str) -> None:
        if not self._callback:
            return
        run_coroutine_threadsafe(self._callback.on_keydown(code), self._loop)

    def update(self, image: NDArray) -> None:
        self.after_threadsafe(remain=0, image=image, milliseconds=0)

    async def until_thread_complete(self) -> None:
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            await self._loop.run_in_executor(executor, self.mainloop)
        except CancelledError:
            self.quit_threadsafe()

            # Wait for the executor to exit ...
            executor.shutdown(wait=True)

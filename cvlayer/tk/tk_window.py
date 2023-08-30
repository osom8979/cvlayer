# -*- coding: utf-8 -*-

from functools import partial
from threading import Lock
from tkinter import NW, Canvas, Event, Tk
from typing import Final, Optional, Tuple

import cv2
from numpy import uint8, zeros
from numpy.typing import NDArray
from PIL.Image import fromarray
from PIL.ImageTk import PhotoImage

INFINITY_AFTER: Final[int] = -1
DEFAULT_WIDTH: Final[int] = 640
DEFAULT_HEIGHT: Final[int] = 360
DEFAULT_X: Final[int] = 0
DEFAULT_Y: Final[int] = 0
DEFAULT_TITLE: Final[str] = "TkWindow"
DEFAULT_FPS: Final[int] = 30
USE_INFINITY_AFTER: Final[bool] = True


class TkWindow:
    _exception: Optional[BaseException]

    def __init__(
        self,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        x=DEFAULT_X,
        y=DEFAULT_Y,
        title=DEFAULT_TITLE,
        fps=DEFAULT_FPS,
        use_infinity_after=USE_INFINITY_AFTER,
    ):
        self._exception = None
        self._milliseconds = 1000 // fps

        self._tk = Tk()
        self._tk.title(title)
        self._tk.geometry(f"{width}x{height}+{x}+{y}")
        self._tk.resizable(True, True)

        self._canvas = Canvas(self._tk, width=width, height=height, bg="white")
        self._canvas.pack(fill="both", expand=True)

        self._image = zeros((width, height, 3), dtype=uint8)
        self._photo = PhotoImage(image=fromarray(self._image, mode="RGB"))
        self._canvas.create_image(0, 0, image=self._photo, anchor=NW)

        self._tk.bind("<Configure>", self._configure)
        self._tk.bind("<Escape>", self._escape)
        self._tk.bind("<Key>", self._key)

        if use_infinity_after:
            self._tk.after(0, partial(self._update, INFINITY_AFTER))
        else:
            self._tk.after(0, partial(self._update, 0))
        self._tk_lock = Lock()

    @property
    def tk(self) -> Tk:
        return self._tk

    @property
    def width(self) -> int:
        return self._tk.winfo_width()

    @property
    def height(self) -> int:
        return self._tk.winfo_height()

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def mainloop(self, shield_exception=False) -> None:
        self._tk.mainloop()  # Holding
        if not shield_exception and self._exception is not None:
            raise RuntimeError from self._exception

    def quit_threadsafe(self) -> None:
        with self._tk_lock:
            self._tk.quit()

    def _configure(self, event: Event) -> None:
        pass

    def _escape(self, event: Event) -> None:
        assert event is not None
        self.on_escape()

    def _key(self, event: Event) -> None:
        assert self._exception is None
        try:
            self.on_keydown(event.char)
        except BaseException as e:
            self._exception = e
            self.quit_threadsafe()

    def cvt_preview(self, image: NDArray) -> NDArray:
        resized = cv2.resize(image, self.size)
        if len(resized.shape) == 2:
            return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            return resized[:, :, ::-1]

    def _update(self, remain=INFINITY_AFTER, image: Optional[NDArray] = None) -> None:
        assert self._exception is None
        try:
            grab = image if image else self.on_grab()
            self._image = self.cvt_preview(grab)
            self._photo = PhotoImage(image=fromarray(self._image, mode="RGB"))
            self._canvas.create_image(0, 0, image=self._photo, anchor=NW)
        except BaseException as e:
            self._exception = e
            self.quit_threadsafe()
        finally:
            if remain == INFINITY_AFTER:
                self.after_threadsafe(INFINITY_AFTER)
            elif remain >= 1:
                self.after_threadsafe(remain - 1)

    def after_threadsafe(
        self,
        remain=INFINITY_AFTER,
        image: Optional[NDArray] = None,
        *,
        milliseconds: Optional[int] = None,
    ) -> None:
        with self._tk_lock:
            ms = milliseconds if milliseconds is not None else self._milliseconds
            self._tk.after(ms, partial(self._update, remain, image))

    def on_grab(self) -> NDArray:
        return zeros((self.width, self.height, 3), dtype=uint8)

    def on_escape(self) -> None:
        self.quit_threadsafe()

    def on_keydown(self, code: str) -> None:
        pass

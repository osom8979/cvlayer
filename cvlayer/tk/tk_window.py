# -*- coding: utf-8 -*-

from tkinter import NW, Canvas, Event, Tk
from typing import Optional, Tuple

import cv2
from numpy import uint8, zeros
from numpy.typing import NDArray
from PIL.Image import fromarray
from PIL.ImageTk import PhotoImage


class TkWindow:
    _exception: Optional[BaseException]

    def __init__(self, width=640, height=360, x=0, y=0, title="TkWindow", fps=30):
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

        self._tk.after(0, self._update)
        self._tk.bind("<Configure>", self._configure)
        self._tk.bind("<Escape>", self._escape)
        self._tk.bind("<Key>", self._key)

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

    def _configure(self, event: Event) -> None:
        pass

    def _escape(self, event: Event) -> None:
        assert event is not None
        self._tk.quit()

    def _key(self, event: Event) -> None:
        assert self._exception is None
        try:
            self.on_keydown(event.char)
        except BaseException as e:
            self._exception = e
            self._tk.quit()

    def _update(self) -> None:
        assert self._exception is None
        try:
            self._image = cv2.resize(self.on_grab(), self.size)[:, :, ::-1]
            self._photo = PhotoImage(image=fromarray(self._image, mode="RGB"))
            self._canvas.create_image(0, 0, image=self._photo, anchor=NW)
        except BaseException as e:
            self._exception = e
            self._tk.quit()
        finally:
            self._tk.after(self._milliseconds, self._update)

    def on_grab(self) -> NDArray:
        return zeros((self.width, self.height, 3), dtype=uint8)

    def on_keydown(self, code: str) -> None:
        pass

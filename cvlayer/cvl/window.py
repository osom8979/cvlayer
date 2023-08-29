# -*- coding: utf-8 -*-

from cvlayer.cv.window import WINDOW_AUTOSIZE, Window


class CvlWindow:
    @staticmethod
    def cvl_create_window(name: str, flags=WINDOW_AUTOSIZE) -> Window:
        return Window(name, flags)

# -*- coding: utf-8 -*-

from cvlayer.cv.backend import highgui_backend, highgui_backend_type


class CvlBackend:
    @staticmethod
    def cvl_highgui_backend():
        return highgui_backend()

    @staticmethod
    def cvl_highgui_backend_type():
        return highgui_backend_type()

# -*- coding: utf-8 -*-

from typing import Union

from cvlayer.cv.video_capture import VideoCapture


class CvlVideoCapture:
    @staticmethod
    def cvl_create_video_capture(file: Union[str, int]) -> VideoCapture:
        return VideoCapture(file)

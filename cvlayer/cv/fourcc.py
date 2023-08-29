# -*- coding: utf-8 -*-

from typing import Callable, Final

import cv2

FourccFuncCallable = Callable[[str, str, str, str], int]
FOURCC_FUNC_NAME = "VideoWriter_fourcc"

assert hasattr(cv2, FOURCC_FUNC_NAME), f"cv2 has no attribute '{FOURCC_FUNC_NAME}'"
fourcc: FourccFuncCallable = getattr(cv2, FOURCC_FUNC_NAME)


def get_fourcc(c1: str, c2: str, c3: str, c4: str) -> int:
    assert len(c1) == 1
    assert len(c2) == 1
    assert len(c3) == 1
    assert len(c4) == 1
    return fourcc(c1, c2, c3, c4)


FOURCC_MP4V: Final[int] = get_fourcc(*"mp4v")
FOURCC_DIVX: Final[int] = get_fourcc(*"divx")  # DivX MPEG-4
FOURCC_XVID: Final[int] = get_fourcc(*"xvid")  # XVID MPEG-4
FOURCC_FMP4: Final[int] = get_fourcc(*"fmp4")  # FFMPEG MPEG4
FOURCC_WMV2: Final[int] = get_fourcc(*"wmv2")  # Windows Media Video 8
FOURCC_MJPG: Final[int] = get_fourcc(*"mjpg")  # Motion JPEG
FOURCC_YV12: Final[int] = get_fourcc(*"yv12")  # YUV 4:2:0 Planar
FOURCC_X264: Final[int] = get_fourcc(*"x264")  # H.264 / AVC
FOURCC_AVC1: Final[int] = get_fourcc(*"avc1")  # Advanced Video

DEFAULT_FOURCC: Final[int] = FOURCC_MP4V


class CvlFourcc:
    @staticmethod
    def cvl_fourcc(c1: str, c2: str, c3: str, c4: str):
        return get_fourcc(c1, c2, c3, c4)

    @staticmethod
    def cvl_fourcc_mp4v():
        return FOURCC_MP4V

    @staticmethod
    def cvl_fourcc_mjpg():
        return FOURCC_MJPG

    @staticmethod
    def cvl_fourcc_with_name(name: str):
        assert len(name) == 4
        return get_fourcc(*name)

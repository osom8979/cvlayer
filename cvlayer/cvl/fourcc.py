# -*- coding: utf-8 -*-

from cvlayer.cv.fourcc import FOURCC_MJPG, FOURCC_MP4V, get_fourcc


class CvlFourcc:
    @staticmethod
    def cvl_fourcc(c1: str, c2: str, c3: str, c4: str) -> int:
        return get_fourcc(c1, c2, c3, c4)

    @staticmethod
    def cvl_fourcc_mp4v() -> int:
        return FOURCC_MP4V

    @staticmethod
    def cvl_fourcc_mjpg() -> int:
        return FOURCC_MJPG

    @staticmethod
    def cvl_fourcc_with_name(name: str) -> int:
        assert len(name) == 4
        return get_fourcc(*name)

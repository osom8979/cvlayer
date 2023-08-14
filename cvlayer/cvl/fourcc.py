# -*- coding: utf-8 -*-

from cvlayer.cv.fourcc import (
    FOURCC_AVC1,
    FOURCC_DIVX,
    FOURCC_FMP4,
    FOURCC_MJPG,
    FOURCC_MP4V,
    FOURCC_WMV2,
    FOURCC_X264,
    FOURCC_XVID,
    FOURCC_YV12,
    get_fourcc,
)


class CvlFourcc:
    FOURCC_MP4V_VALUE = FOURCC_MP4V
    FOURCC_DIVX_VALUE = FOURCC_DIVX
    FOURCC_XVID_VALUE = FOURCC_XVID
    FOURCC_FMP4_VALUE = FOURCC_FMP4
    FOURCC_WMV2_VALUE = FOURCC_WMV2
    FOURCC_MJPG_VALUE = FOURCC_MJPG
    FOURCC_YV12_VALUE = FOURCC_YV12
    FOURCC_X264_VALUE = FOURCC_X264
    FOURCC_AVC1_VALUE = FOURCC_AVC1

    @staticmethod
    def cvl_fourcc(c1: str, c2: str, c3: str, c4: str) -> int:
        return get_fourcc(c1, c2, c3, c4)

    @staticmethod
    def cvl_fourcc_with_name(name: str) -> int:
        assert len(name) == 4
        return get_fourcc(*name)

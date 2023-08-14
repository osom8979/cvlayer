# -*- coding: utf-8 -*-

from cvlayer.cv.cvt_color import CvtColorCode, cvt_color
from cvlayer.types import Image


class CvlCvtColor:
    CvtColorCodeType = CvtColorCode

    @staticmethod
    def cvl_cvt_color(image: Image, code: CvtColorCode) -> Image:
        return cvt_color(image, code)

    @staticmethod
    def cvl_cvt_color_bgr2gray(image: Image) -> Image:
        return CvlCvtColor.cvl_cvt_color(image, CvtColorCode.BGR2GRAY)

    @staticmethod
    def cvl_cvt_color_gray2bgr(image: Image) -> Image:
        return CvlCvtColor.cvl_cvt_color(image, CvtColorCode.GRAY2BGR)

    @staticmethod
    def cvl_cvt_color_bgr2hsv(image: Image) -> Image:
        return CvlCvtColor.cvl_cvt_color(image, CvtColorCode.BGR2HSV)

    @staticmethod
    def cvl_cvt_color_hsv2bgr(image: Image) -> Image:
        return CvlCvtColor.cvl_cvt_color(image, CvtColorCode.HSV2BGR)

    @staticmethod
    def cvl_cvt_color_bgr2yuv(image: Image) -> Image:
        return CvlCvtColor.cvl_cvt_color(image, CvtColorCode.BGR2YUV)

    @staticmethod
    def cvl_cvt_color_yuv2bgr(image: Image) -> Image:
        return CvlCvtColor.cvl_cvt_color(image, CvtColorCode.YUV2BGR)

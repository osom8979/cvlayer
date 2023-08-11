# -*- coding: utf-8 -*-

from enum import Enum, unique

import cv2

from cvlayer.types.np import ImageType


@unique
class CvtColorCode(Enum):
    BGR2GRAY = cv2.COLOR_BGR2GRAY
    GRAY2BGR = cv2.COLOR_GRAY2BGR

    BGR2HSV = cv2.COLOR_BGR2HSV
    HSV2BGR = cv2.COLOR_HSV2BGR

    BGR2YUV = cv2.COLOR_BGR2YUV
    YUV2BGR = cv2.COLOR_YUV2BGR


def cvt_color(image: ImageType, code: CvtColorCode) -> ImageType:
    return cv2.cvtColor(image, code.value)


class CvlCvtColor:
    CvtColorCodeType = CvtColorCode

    @staticmethod
    def cvl_cvt_color(image: ImageType, code: CvtColorCode) -> ImageType:
        return cvt_color(image, code)

    @staticmethod
    def cvl_cvt_color_bgr2gray(image: ImageType) -> ImageType:
        return CvlCvtColor.cvl_cvt_color(image, CvtColorCode.BGR2GRAY)

    @staticmethod
    def cvl_cvt_color_gray2bgr(image: ImageType) -> ImageType:
        return CvlCvtColor.cvl_cvt_color(image, CvtColorCode.GRAY2BGR)

    @staticmethod
    def cvl_cvt_color_bgr2hsv(image: ImageType) -> ImageType:
        return CvlCvtColor.cvl_cvt_color(image, CvtColorCode.BGR2HSV)

    @staticmethod
    def cvl_cvt_color_hsv2bgr(image: ImageType) -> ImageType:
        return CvlCvtColor.cvl_cvt_color(image, CvtColorCode.HSV2BGR)

    @staticmethod
    def cvl_cvt_color_bgr2yuv(image: ImageType) -> ImageType:
        return CvlCvtColor.cvl_cvt_color(image, CvtColorCode.BGR2YUV)

    @staticmethod
    def cvl_cvt_color_yuv2bgr(image: ImageType) -> ImageType:
        return CvlCvtColor.cvl_cvt_color(image, CvtColorCode.YUV2BGR)

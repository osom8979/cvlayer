# -*- coding: utf-8 -*-

from enum import Enum, unique

import cv2

from cvlayer.types.np import Image


@unique
class CvtColorCode(Enum):
    BGR2GRAY = cv2.COLOR_BGR2GRAY
    GRAY2BGR = cv2.COLOR_GRAY2BGR

    BGR2HSV = cv2.COLOR_BGR2HSV
    HSV2BGR = cv2.COLOR_HSV2BGR

    BGR2YUV = cv2.COLOR_BGR2YUV
    YUV2BGR = cv2.COLOR_YUV2BGR


def cvt_color(image: Image, code: CvtColorCode) -> Image:
    return cv2.cvtColor(image, code.value)

# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final

import cv2

CV_8U: Final[int] = cv2.CV_8U  # type: ignore[attr-defined]
CV_8S: Final[int] = cv2.CV_8S  # type: ignore[attr-defined]
assert CV_8U == 0
assert CV_8S == 1

CV_16U: Final[int] = cv2.CV_16U  # type: ignore[attr-defined]
CV_16S: Final[int] = cv2.CV_16S  # type: ignore[attr-defined]
assert CV_16U == 2
assert CV_16S == 3

assert not hasattr(cv2, "CV_32U")
CV_32S: Final[int] = cv2.CV_32S  # type: ignore[attr-defined]
assert CV_32S == 4

assert not hasattr(cv2, "CV_64U")
assert not hasattr(cv2, "CV_64S")

CV_32F: Final[int] = cv2.CV_32F  # type: ignore[attr-defined]
CV_64F: Final[int] = cv2.CV_64F  # type: ignore[attr-defined]
assert CV_32F == 5
assert CV_64F == 6

CV_16F: Final[int] = cv2.CV_16F  # type: ignore[attr-defined]
assert CV_16F == 7


@unique
class CvDataType(Enum):
    U8 = CV_8U
    S8 = CV_8S
    U16 = CV_16U
    S16 = CV_16S
    S32 = CV_32S
    F32 = CV_32F
    F64 = CV_64F
    F16 = CV_16F

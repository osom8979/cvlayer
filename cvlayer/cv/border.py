# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final

import cv2


@unique
class BorderType(Enum):
    CONSTANT = cv2.BORDER_CONSTANT
    REPLICATE = cv2.BORDER_REPLICATE
    REFLECT = cv2.BORDER_REFLECT
    # BORDER = cv2.BORDER_WRAP  # BORDER_WRAP is not supported
    REFLECT101 = cv2.BORDER_REFLECT101
    TRANSPARENT = cv2.BORDER_TRANSPARENT
    ISOLATED = cv2.BORDER_ISOLATED


DEFAULT_BORDER_TYPE: Final[BorderType] = BorderType.REFLECT101

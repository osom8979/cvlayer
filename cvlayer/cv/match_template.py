# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import NamedTuple, Optional

import cv2
from numpy.typing import NDArray


@unique
class MatchTemplateMethod(Enum):
    SQDIFF = cv2.TM_SQDIFF
    SQDIFF_NORMED = cv2.TM_SQDIFF_NORMED
    CCORR = cv2.TM_CCORR
    CCORR_NORMED = cv2.TM_CCORR_NORMED
    CCOEFF = cv2.TM_CCOEFF
    CCOEFF_NORMED = cv2.TM_CCOEFF_NORMED


class MatchResult(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


def match_template(
    src: NDArray,
    template: NDArray,
    method=MatchTemplateMethod.SQDIFF_NORMED,
    mask: Optional[NDArray] = None,
) -> MatchResult:
    # https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html
    res = cv2.matchTemplate(src, template, method.value, mask=mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res, mask=mask)

    if method in (MatchTemplateMethod.SQDIFF, MatchTemplateMethod.SQDIFF_NORMED):
        top_left = min_loc  # min is good matching
        match_val = min_val
    else:
        top_left = max_loc  # max is good matching
        match_val = max_val

    template_height, template_width = template.shape[:2]
    x1 = top_left[0]
    y1 = top_left[1]
    x2 = x1 + template_width
    y2 = y1 + template_height

    return MatchResult(x1, y1, x2, y2, match_val)


class CvlMatchTemplate:
    @staticmethod
    def cvl_match_template(
        src: NDArray,
        template: NDArray,
        method=MatchTemplateMethod.SQDIFF_NORMED,
        mask: Optional[NDArray] = None,
    ):
        return match_template(src, template, method, mask)

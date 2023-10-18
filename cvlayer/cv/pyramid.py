# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

from cvlayer.cv.term_criteria import TermCriteria, TermCriteriaType

DEFAULT_SPATIAL_WINDOW_RADIUS: Final[float] = 20.0
DEFAULT_COLOR_WINDOW_RADIUS: Final[float] = 20.0
DEFAULT_MAX_LEVEL: Final[int] = 1

DEFAULT_TERM_CRITERIA_TYPE: Final[TermCriteriaType] = TermCriteriaType.COUNT_EPS
DEFAULT_TERM_CRITERIA_MAX_COUNT: Final[int] = 5
DEFAULT_TERM_CRITERIA_EPSILON: Final[float] = 1.0

DEFAULT_TERM_CRITERIA: Final[TermCriteria] = TermCriteria(
    DEFAULT_TERM_CRITERIA_TYPE,
    DEFAULT_TERM_CRITERIA_MAX_COUNT,
    DEFAULT_TERM_CRITERIA_EPSILON,
)


def pyr_mean_shift_filtering(
    src: NDArray,
    sp=DEFAULT_SPATIAL_WINDOW_RADIUS,
    sr=DEFAULT_COLOR_WINDOW_RADIUS,
    max_level=DEFAULT_MAX_LEVEL,
    term_criteria=DEFAULT_TERM_CRITERIA,
) -> NDArray:
    criteria = (
        term_criteria.type.value,
        term_criteria.max_count,
        term_criteria.epsilon,
    )
    return cv2.pyrMeanShiftFiltering(
        src=src,
        sp=sp,
        sr=sr,
        dst=None,
        maxLevel=max_level,
        termcrit=criteria,
    )


class CvlPyramid:
    @staticmethod
    def cvl_pyr_mean_shift_filtering(
        src: NDArray,
        sp=DEFAULT_SPATIAL_WINDOW_RADIUS,
        sr=DEFAULT_COLOR_WINDOW_RADIUS,
        max_level=DEFAULT_MAX_LEVEL,
        term_criteria=DEFAULT_TERM_CRITERIA,
    ):
        pyr_mean_shift_filtering(src, sp, sr, max_level, term_criteria)

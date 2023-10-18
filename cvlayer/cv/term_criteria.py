# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import NamedTuple

import cv2

assert cv2.TERM_CRITERIA_COUNT == cv2.TERM_CRITERIA_MAX_ITER


@unique
class TermCriteriaType(Enum):
    COUNT = cv2.TERM_CRITERIA_COUNT
    """
    the maximum number of iterations or elements to compute
    """

    EPS = cv2.TERM_CRITERIA_EPS
    """
    the desired accuracy or change in parameters at which the iterative algorithm stops
    """

    COUNT_EPS = cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS
    """
    COUNT + EPS
    """


class TermCriteria(NamedTuple):
    type: TermCriteriaType
    """
    The type of termination criteria: COUNT, EPS or COUNT + EPS
    """

    max_count: int
    """
    The maximum number of iterations/elements
    """

    epsilon: float
    """
    The desired accuracy
    """

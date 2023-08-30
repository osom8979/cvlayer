# -*- coding: utf-8 -*-

from enum import Enum, auto, unique
from typing import Final

import cv2
from numpy.typing import NDArray

DEFAULT_K: Final[int] = 2
DEFAULT_NORM_TYPE = None
DEFAULT_CROSS_CHECK = None


@unique
class MatcherType(Enum):
    BF = auto()


class Matcher:
    def __init__(
        self,
        matcher=MatcherType.BF,
        norm_type=DEFAULT_NORM_TYPE,
        cross_check=DEFAULT_CROSS_CHECK,
    ):
        if matcher == MatcherType.BF:
            self._matcher = cv2.BFMatcher(normType=norm_type, crossCheck=cross_check)
        else:
            raise NotImplementedError

    def match(self, desc1: NDArray, desc2: NDArray, k=DEFAULT_K):
        return self._matcher.knnMatch(desc1, desc2, k=k)


class CvlMatcher:
    @staticmethod
    def cvl_create_matcher(
        matcher=MatcherType.BF,
        norm_type=DEFAULT_NORM_TYPE,
        cross_check=DEFAULT_CROSS_CHECK,
    ):
        return Matcher(matcher, norm_type, cross_check)

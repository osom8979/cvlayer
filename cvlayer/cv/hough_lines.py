# -*- coding: utf-8 -*-

from math import pi
from typing import Final

import cv2
from numpy.typing import NDArray

DEFAULT_RHO: Final[float] = 1.0
"""Distance resolution of the accumulator in pixels."""

DEFAULT_THETA: Final[float] = pi / 180.0
"""Angle resolution of the accumulator in radians."""

DEFAULT_THRESHOLD: Final[int] = 100
"""Accumulator threshold parameter."""


def hough_lines(
    frame: NDArray,
    rho=DEFAULT_RHO,
    theta=DEFAULT_THETA,
    threshold=DEFAULT_THRESHOLD,
) -> NDArray:
    return cv2.HoughLines(frame, rho, theta, threshold)


class CvlHoughLines:
    @staticmethod
    def cvl_hough_lines(
        frame: NDArray,
        rho=DEFAULT_RHO,
        theta=DEFAULT_THETA,
        threshold=DEFAULT_THRESHOLD,
    ) -> NDArray:
        return hough_lines(frame, rho, theta, threshold)

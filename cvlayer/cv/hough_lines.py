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
    image: NDArray,
    rho=DEFAULT_RHO,
    theta=DEFAULT_THETA,
    threshold=DEFAULT_THRESHOLD,
    srn=0.0,
    stn=0.0,
    min_theta=0.0,
    max_theta=pi,
) -> NDArray:
    """Finds lines in a binary image using the standard Hough transform.

    :param image: 8-bit, single-channel binary source image.
        The image may be modified by the function.
    :param rho: Distance resolution of the accumulator in pixels.
    :param theta: Angle resolution of the accumulator in radians.
    :param threshold: Accumulator threshold parameter.
        Only those lines are returned that get enough votes ( >threshold ).
    :param srn: For the multiscale Hough transform,
        it is a divisor for the distance resolution rho.
        The coarse accumulator distance resolution is rho and the accurate accumulator
        resolution is rho/srn. If both srn=0 and stn=0, the classical Hough transform
        is used. Otherwise, both these parameters should be positive.
    :param stn: For the multiscale Hough transform,
        it is a divisor for the distance resolution theta.
    :param min_theta: For standard and multiscale Hough transform,
        minimum angle to check for lines. Must fall between 0 and max_theta.
    :param max_theta: For standard and multiscale Hough transform,
        an upper bound for the angle. Must fall between min_theta and CV_PI.
        The actual maximum angle in the accumulator may be slightly less than max_theta,
        depending on the parameters min_theta and theta.

    :return: Output vector of lines.
        Each line is represented by a 2 or 3 element vector (ρ,θ) or (ρ,θ,votes),
        where ρ is the distance from the coordinate origin (0,0)
        (top-left corner of the image),
        θ is the line rotation angle in radians ( 0∼vertical line,π/2∼horizontal line ),
        and votes is the value of accumulator.
    """
    assert len(image.shape) == 2
    assert image.dtype.itemsize == 1
    return cv2.HoughLines(
        image, rho, theta, threshold, None, srn, stn, min_theta, max_theta
    )


class CvlHoughLines:
    @staticmethod
    def cvl_hough_lines(
        image: NDArray,
        rho=DEFAULT_RHO,
        theta=DEFAULT_THETA,
        threshold=DEFAULT_THRESHOLD,
        srn=0.0,
        stn=0.0,
        min_theta=0.0,
        max_theta=pi,
    ) -> NDArray:
        return hough_lines(image, rho, theta, threshold, srn, stn, min_theta, max_theta)

# -*- coding: utf-8 -*-

from math import atan2, degrees
from typing import overload

from cvlayer.math.constant import DOUBLE_PI
from cvlayer.typing import PointT


def get_sign_value(value) -> int:
    return 1 if value >= 0 else -1


# fmt: off
@overload
def normalize_degrees_360(angle: int) -> int: ...
@overload
def normalize_degrees_360(angle: float) -> float: ...
# fmt: on


def normalize_degrees_360(angle):
    result = angle % 360
    assert 0 <= result < 360
    return result


# fmt: off
@overload
def normalize_signed_degrees_360(angle: int) -> int: ...
@overload
def normalize_signed_degrees_360(angle: float) -> float: ...
# fmt: on


def normalize_signed_degrees_360(angle):
    """
    Used when sign information must be maintained.
    """

    # [WARNING] Reasons for recalculating with get_sign_value:
    # `-180 // 360 = -1`
    # `-180 % 360 = 180`
    _sign = get_sign_value(angle)
    result = angle - (abs(angle) // 360 * 360 * _sign)
    assert -360 < result < 360
    return result


def radians_angle(x: float, y: float) -> float:
    theta = atan2(y, x)
    result = theta if theta >= 0 else DOUBLE_PI + theta
    assert 0 <= result < DOUBLE_PI
    return result


def degrees_angle(x: float, y: float) -> float:
    return degrees(radians_angle(x, y))


def radians_point1(p: PointT) -> float:
    return radians_angle(p[0], p[1])


def degrees_point1(p: PointT) -> float:
    return degrees(radians_point1(p))


def radians_point2(p1: PointT, p2: PointT) -> float:
    return radians_angle(p2[0] - p1[0], p2[1] - p1[1])


def degrees_point2(p1: PointT, p2: PointT) -> float:
    return degrees(radians_point2(p1, p2))


def radians_point3(a: PointT, b: PointT, c: PointT) -> float:
    ##############
    #       a    #
    #      *     #
    #     /      #
    #    /       #
    # b *----* c #
    ##############
    a2b = radians_angle(a[0] - b[0], a[1] - b[1])
    c2b = radians_angle(c[0] - b[0], c[1] - b[1])
    a2b = DOUBLE_PI if a2b == 0 else a2b
    c2b = DOUBLE_PI if c2b == 0 else c2b
    assert 0 < a2b <= DOUBLE_PI
    assert 0 < c2b <= DOUBLE_PI
    result = abs((c2b - a2b) % DOUBLE_PI)
    assert 0 <= result < DOUBLE_PI
    return result


def degrees_point3(a: PointT, b: PointT, c: PointT) -> float:
    return degrees(radians_point3(a, b, c))


def clockwise_radians_point3(a: PointT, b: PointT, c: PointT) -> float:
    result = (DOUBLE_PI - radians_point3(a, b, c)) % DOUBLE_PI
    assert 0 <= result < DOUBLE_PI
    return result


def clockwise_degrees_point3(a: PointT, b: PointT, c: PointT) -> float:
    return degrees(clockwise_radians_point3(a, b, c))

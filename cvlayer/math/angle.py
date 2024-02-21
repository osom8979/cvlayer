# -*- coding: utf-8 -*-

from math import atan2, degrees
from typing import Literal, overload

from cvlayer.cv.types.shape import PointT
from cvlayer.math.constant import DOUBLE_PI, PI


def get_sign(value) -> Literal[-1, 1]:
    return -1 if value < 0 else 1


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

    # [IMPORTANT] Reasons for recalculating with 'get_sign_value':
    # `-180 // 360 = -1`
    # `-180 % 360 = 180`
    sign = get_sign(angle)

    number_of_rotations = abs(angle) // 360
    total_rotation_angle = number_of_rotations * 360
    total_rotation_angle_with_sign = total_rotation_angle * sign

    result = angle - total_rotation_angle_with_sign
    assert -360 < result < 360
    return result


def normalize_signed_degrees_180(angle: float) -> float:
    next_angle = normalize_signed_degrees_360(angle)
    assert -360 < next_angle < 360

    if -360 < next_angle <= -180:
        return next_angle + 360
    elif -180 < next_angle <= 180:
        return next_angle
    elif 180 < next_angle < 360:
        return next_angle - 360
    else:
        assert False, "Inaccessible section"


def radians_angle(x: float, y: float) -> float:
    theta = atan2(y, x)  # Return the arc tangent of y/x in radians
    assert -PI <= theta <= PI

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
    result = degrees(radians_point3(a, b, c))
    assert 0 <= result < 360
    return result


def clockwise_radians_point3(a: PointT, b: PointT, c: PointT) -> float:
    result = (DOUBLE_PI - radians_point3(a, b, c)) % DOUBLE_PI
    assert 0 <= result < DOUBLE_PI
    return result


def clockwise_degrees_point3(a: PointT, b: PointT, c: PointT) -> float:
    result = degrees(clockwise_radians_point3(a, b, c))
    assert 0 <= result < 360
    return result


def normalize_point(point: PointT, center: PointT) -> PointT:
    """
    Move the center coordinate to the origin coordinate.
    """

    px, py = point
    cx, cy = center
    return px - cx, py - cy


def measure_angle_diff(
    point0: PointT,
    center0: PointT,
    point1: PointT,
    center1: PointT,
) -> float:
    a = normalize_point(point0, center0)
    c = normalize_point(point1, center1)
    return degrees_point3(a, (0.0, 0.0), c)


def close_to_pivot(angle: float, pivot: float, delta: float) -> bool:
    """
    Make sure the angle presented is close to the pivot angle.
    """

    if delta < 0:
        raise ValueError("The 'delta' value must be greater than or equal to 0")

    return abs(angle - pivot) <= abs(delta)

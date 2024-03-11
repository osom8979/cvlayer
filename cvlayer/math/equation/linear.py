# -*- coding: utf-8 -*-

from enum import Enum, unique
from math import atan, cos
from math import degrees as math_degrees
from math import pi, sin, sqrt, tan


class ParallelError(Exception):
    pass


@unique
class RelativePosition(Enum):
    LEFT = -1
    ON = 0
    RIGHT = 1


class SlopeInterceptForm:
    """
    Equation of a Straight Line

    y = mx + b
    """

    def __init__(self, m, b):
        self.m = m
        self.b = b


class PointSlopeForm:
    """
    Point-Slope Equation of a Line

    y − y1 = m(x − x1)
    """

    def __init__(self, x1, y1, m):
        self.x1 = x1
        self.y1 = y1
        self.m = m


class GeneralForm:
    """
    General Form of Equation of a Line

    Ax + By + C = 0
    """

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    @classmethod
    def from_coord_points(cls, x1, y1, x2, y2):
        """
        (y1 - y2)x + (x2 - x1)y + (x1y2 - x2y1) = 0
        """
        a = y1 - y2
        b = x2 - x1
        c = (x1 * y2) - (x2 * y1)
        return cls(a, b, c)

    @classmethod
    def from_points(cls, p1, p2):
        return cls.from_coord_points(p1[0], p1[1], p2[0], p2[1])

    @classmethod
    def from_coord_polar(cls, x1, y1, radians):
        """
        m = tan(radians)
        y - y1 = m (x - x1)
        y = mx - mx1 + y1
        0 = mx - mx1 + y1 - y
        0 = mx - y + (-mx1 + y1)
        A = m
        B = -1
        C = -m * x1 + y1
        """
        m = tan(radians)
        a = m
        b = -1
        c = -m * x1 + y1
        return cls(a, b, c)

    @classmethod
    def from_distance_polar(cls, distance, radians):
        x = distance * cos(radians)
        y = distance * sin(radians)
        return cls.from_coord_polar(x, y, radians + pi / 2)

    @classmethod
    def from_polar(cls, p1, radians):
        return cls.from_coord_polar(p1[0], p1[1], radians)

    @classmethod
    def from_coord_degrees_polar(cls, x1, y1, degrees):
        """
        radians = degrees / 180 * pi
        """
        return cls.from_coord_polar(x1, y1, degrees / 180 * pi)

    @classmethod
    def from_degrees_polar(cls, p1, degrees):
        """
        radians = degrees / 180 * pi
        """
        return cls.from_coord_polar(p1[0], p1[1], degrees / 180 * pi)

    def valid_coord(self, x, y):
        return self.a * x + self.b * y + self.c

    def valid(self, point):
        return self.valid_coord(point[0], point[1])

    def calc_x(self, y):
        """
        Ax + By + C = 0
        Ax = -By -C
        Ax = -(By + C)
        x = -(By + C) / A
        """
        return -1 * ((self.b * y) + self.c) / self.a

    def calc_y(self, x):
        """
        Ax + By + C = 0
        By = -Ax -C
        By = -(Ax + C)
        y = -(Ax + C) / B
        """
        return -1 * ((self.a * x) + self.c) / self.b

    @property
    def slope(self):
        """
        Slope of linear equation
        """
        return -1 * (self.a / self.b)

    @property
    def radians(self):
        return atan(self.slope)

    @property
    def degrees(self):
        return math_degrees(self.radians)

    @property
    def unsigned_radians(self):
        theta = self.radians
        assert -pi <= theta <= pi

        result = theta if theta >= 0 else 2 * pi + theta
        assert 0 <= result < 2 * pi
        return result

    @property
    def unsigned_degrees(self):
        return math_degrees(self.unsigned_radians)

    @property
    def vertical_slope(self):
        """
        vertical_slope * slope = -1
        vertical_slope = -1 / slope
        vertical_slope = -1 / (-1 * (self.a / self.b))
        vertical_slope = -1 * -1 * (self.b / self.a)
        vertical_slope = self.b / self.a
        """
        return self.b / self.a

    @property
    def vertical_radians(self):
        return atan(self.vertical_slope)

    @property
    def vertical_degrees(self):
        return math_degrees(self.vertical_radians)

    def create_vertical_form(self, x3, y3):
        return type(self).from_coord_polar(x3, y3, self.vertical_radians)

    def _get_drawable_left(self, canvas_left):
        if self.b == 0:
            return None
        return canvas_left, self.calc_y(canvas_left)

    def _get_drawable_top(self, canvas_top):
        if self.a == 0:
            return None
        return self.calc_x(canvas_top), canvas_top

    def _get_drawable_right(self, canvas_right):
        if self.b == 0:
            return None
        return canvas_right, self.calc_y(canvas_right)

    def _get_drawable_bottom(self, canvas_bottom):
        if self.a == 0:
            return None
        return self.calc_x(canvas_bottom), canvas_bottom

    def get_drawable_points(self, canvas_roi):
        canvas_left, canvas_top, canvas_right, canvas_bottom = canvas_roi

        left = self._get_drawable_left(canvas_left)
        top = self._get_drawable_top(canvas_top)
        right = self._get_drawable_right(canvas_right)
        bottom = self._get_drawable_bottom(canvas_bottom)

        points = set()
        if left is not None and canvas_top <= left[1] <= canvas_bottom:
            points.add(left)
        if top is not None and canvas_left <= top[0] <= canvas_right:
            points.add(top)
        if right is not None and canvas_top <= right[1] <= canvas_bottom:
            points.add(right)
        if bottom is not None and canvas_left <= bottom[0] <= canvas_right:
            points.add(bottom)

        if not points:
            raise IndexError("Out of canvas range error")

        assert len(points) in (1, 2)

        if len(points) == 1:
            p0 = points.pop()
            return p0, p0
        else:
            assert len(points) == 2
            p1 = points.pop()
            p2 = points.pop()
            return p1, p2

    def intersection(self, other: "GeneralForm"):
        """
        Ax + By + C = 0
        Dx + Ey + F = 0
        x = (CE - BF) / (BD - AE)
        y = (CD - AF) / (AE - BD)
        """

        a = self.a
        b = self.b
        c = self.c

        d = other.a
        e = other.b
        f = other.c

        x_numerator = c * e - b * f
        x_denominator = b * d - a * e

        y_numerator = c * d - a * f
        y_denominator = a * e - b * d

        if x_denominator == 0 or y_denominator == 0:
            raise ParallelError("Two straight lines are parallel or coincident")

        return x_numerator / x_denominator, y_numerator / y_denominator

    def distance_to_coord_point(self, x, y):
        """
        d = |ax + by + c| / sqrt(a^2 + b^2)
        """
        return abs(x * self.a + y * self.b + self.c) / sqrt(self.a**2 + self.b**2)

    def distance_to_point(self, point):
        return self.distance_to_coord_point(point[0], point[1])

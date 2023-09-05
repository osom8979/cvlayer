# -*- coding: utf-8 -*-

from cvlayer.math.angle import (
    clockwise_degrees_point3,
    clockwise_radians_point3,
    degrees_angle,
    degrees_point1,
    degrees_point2,
    degrees_point3,
    radians_angle,
    radians_point1,
    radians_point2,
    radians_point3,
)
from cvlayer.typing import PointT


class CvlAngle:
    @staticmethod
    def cvl_radians_angle(x: float, y: float):
        return radians_angle(x, y)

    @staticmethod
    def cvl_degrees_angle(x: float, y: float):
        return degrees_angle(x, y)

    @staticmethod
    def cvl_radians_point1(p: PointT):
        return radians_point1(p)

    @staticmethod
    def cvl_degrees_point1(p: PointT):
        return degrees_point1(p)

    @staticmethod
    def cvl_radians_point2(p1: PointT, p2: PointT):
        return radians_point2(p1, p2)

    @staticmethod
    def cvl_degrees_point2(p1: PointT, p2: PointT):
        return degrees_point2(p1, p2)

    @staticmethod
    def cvl_radians_point3(a: PointT, b: PointT, c: PointT):
        return radians_point3(a, b, c)

    @staticmethod
    def cvl_degrees_point3(a: PointT, b: PointT, c: PointT):
        return degrees_point3(a, b, c)

    @staticmethod
    def cvl_clockwise_radians_point3(a: PointT, b: PointT, c: PointT):
        return clockwise_radians_point3(a, b, c)

    @staticmethod
    def cvl_clockwise_degrees_point3(a: PointT, b: PointT, c: PointT):
        return clockwise_degrees_point3(a, b, c)

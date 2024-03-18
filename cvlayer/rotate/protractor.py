# -*- coding: utf-8 -*-

from typing import Optional

from cvlayer.cv.types.shape import PointT
from cvlayer.math.angle import degrees_point3


class Protractor:
    def __init__(
        self,
        first_center: Optional[PointT] = None,
        first_point: Optional[PointT] = None,
        second_center: Optional[PointT] = None,
        second_point: Optional[PointT] = None,
    ):
        self.first_center = first_center
        self.first_point = first_point
        self.second_center = second_center
        self.second_point = second_point

    @property
    def has_first(self) -> bool:
        return self.first_center is not None and self.first_point is not None

    @property
    def has_second(self) -> bool:
        return self.second_center is not None and self.second_point is not None

    @property
    def exists(self) -> bool:
        return self.has_first and self.has_second

    @staticmethod
    def normalize_point(point: PointT, center: PointT) -> PointT:
        """
        Move the center coordinate to the origin coordinate.
        """

        px, py = point
        cx, cy = center
        return px - cx, py - cy

    @property
    def normalize_first(self) -> PointT:
        if not self.has_first:
            raise ValueError("The first center/point is not ready")

        assert self.first_center is not None
        assert self.first_point is not None

        return self.normalize_point(self.first_point, self.first_center)

    @property
    def normalize_second(self) -> PointT:
        if not self.has_second:
            raise ValueError("The second center/point is not ready")

        assert self.second_center is not None
        assert self.second_point is not None

        return self.normalize_point(self.second_point, self.second_center)

    @property
    def angle(self) -> float:
        result = degrees_point3(
            a=self.normalize_first,
            b=(0.0, 0.0),
            c=self.normalize_second,
        )
        assert 0 <= result < 360
        return result

    @property
    def signed_angle(self) -> float:
        angle = self.angle
        assert 0 <= angle < 360
        result = angle if angle <= 180 else angle - 360
        assert -180 < result <= 180
        return result

    def clear_first(self) -> None:
        self.first_center = None
        self.first_point = None

    def clear_second(self) -> None:
        self.second_center = None
        self.second_point = None

    def clear(self) -> None:
        self.clear_first()
        self.clear_second()

    def set_first(self, center: PointT, point: PointT) -> None:
        self.first_center = center
        self.first_point = point

    def set_second(self, center: PointT, point: PointT) -> None:
        self.second_center = center
        self.second_point = point

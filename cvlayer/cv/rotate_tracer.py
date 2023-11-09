# -*- coding: utf-8 -*-

from typing import Final, Optional, Tuple

import cv2

from cvlayer.geometry.find_nearset_point import find_nearest_point
from cvlayer.math.angle import degrees_point3
from cvlayer.typing import PointT, PolygonT

DEFAULT_MAX_MISSING_COUNT: Final[int] = 10


def normalize_point(pivot: PointT, center: PointT) -> PointT:
    x1, y1 = pivot
    x2, y2 = center
    return x2 - x1, y2 - y1


def calc_degrees(
    prev_point0: PointT,
    prev_center: PointT,
    next_point0: PointT,
    next_center: PointT,
) -> float:
    n0 = normalize_point(prev_point0, prev_center)
    n1 = normalize_point(next_point0, next_center)
    return degrees_point3(n0, (0.0, 0.0), n1)


def measure_center_point(contour) -> PointT:
    m = cv2.moments(contour)
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return cx, cy


class RotateTracer:
    _first_polygon: Optional[PolygonT]
    _first_center: Optional[PointT]
    _first_point0: Optional[PointT]

    _current_polygon: Optional[PolygonT]
    _current_center: Optional[PointT]
    _current_point0: Optional[PointT]

    def __init__(self, max_missing_count=DEFAULT_MAX_MISSING_COUNT):
        self._max_missing_count = max_missing_count

        self._missing_count = 0
        self._weighted_rotate = 0.0
        self._current_rotate = 0.0

        self._first_polygon = None
        self._first_center = None
        self._first_point0 = None

        self._current_polygon = None
        self._current_center = None
        self._current_point0 = None

    @property
    def has_first_polygon(self) -> bool:
        return self._first_polygon is not None

    @property
    def overflow_missing_count(self) -> bool:
        return self._missing_count >= self._max_missing_count

    @property
    def max_missing_count(self) -> int:
        return self._max_missing_count

    @max_missing_count.setter
    def max_missing_count(self, value: int) -> None:
        self._max_missing_count = value

    @property
    def missing_count(self) -> int:
        return self._missing_count

    @property
    def current_center(self) -> Optional[Tuple[float, float]]:
        return self._current_center

    @property
    def current_point0(self) -> Optional[Tuple[float, float]]:
        return self._current_point0

    @property
    def current_rotate_delta(self) -> float:
        current_rotate = self._current_rotate
        if current_rotate > 180:
            return current_rotate - 360
        else:
            return current_rotate

    @property
    def rotate_degrees(self) -> float:
        return self.current_rotate_delta + self._weighted_rotate

    def clear(self) -> None:
        self._missing_count = 0
        self._weighted_rotate = 0.0
        self._current_rotate = 0.0

        self._first_polygon = None
        self._first_center = None
        self._first_point0 = None

        self._current_polygon = None
        self._current_center = None
        self._current_point0 = None

    def reset_missing_count(self) -> None:
        self._missing_count = 0

    def increase_missing_count(self) -> None:
        self._missing_count += 1

    def do_missing(self) -> None:
        if not self.has_first_polygon:
            return

        self.increase_missing_count()

        if self.overflow_missing_count:
            self.clear()

    def do_trace(self, polygon: PolygonT, center: PointT) -> None:
        self.reset_missing_count()
        if self.has_first_polygon:
            self.on_trace_next_angle(polygon, center)
        else:
            self.on_trace_first_angle(polygon, center)

    def on_trace_first_angle(self, polygon: PolygonT, center: PointT) -> None:
        assert polygon
        assert center
        assert isinstance(polygon, list)
        assert isinstance(center, tuple)

        self._weighted_rotate = 0.0
        self._current_rotate = 0.0

        self._first_polygon = polygon
        self._first_center = center
        self._first_point0 = (polygon[0][0], polygon[0][1])

        self._current_polygon = self._first_polygon
        self._current_center = self._first_center
        self._current_point0 = self._first_point0

    def on_trace_next_angle(self, polygon: PolygonT, center: PointT) -> None:
        assert polygon
        assert center
        assert isinstance(polygon, list)
        assert isinstance(center, tuple)

        first_point0 = self._first_point0
        first_center = self._first_center
        current_point0 = self._current_point0
        assert first_point0 is not None
        assert first_center is not None
        assert current_point0 is not None

        next_point0 = find_nearest_point(current_point0, *polygon)
        next_center = center
        next_degrees = calc_degrees(
            first_point0,
            first_center,
            next_point0,
            next_center,
        )
        self._current_polygon = polygon
        self._current_center = next_center
        self._current_point0 = next_point0
        self._current_rotate = next_degrees

    def run(
        self,
        detect: bool,
        polygon: Optional[PolygonT] = None,
        center: Optional[PointT] = None,
    ) -> None:
        if detect:
            assert polygon
            assert center
            assert isinstance(polygon, list)
            assert isinstance(center, tuple)
            self.do_trace(polygon, center)
        else:
            self.do_missing()


class CvlRotateTracer:
    @staticmethod
    def cvl_create_rotate_tracer(max_missing_count=DEFAULT_MAX_MISSING_COUNT):
        return RotateTracer(max_missing_count)

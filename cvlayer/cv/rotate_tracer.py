# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Final, Optional, Tuple

import cv2

from cvlayer.geometry.find_nearset_point import find_nearest_point
from cvlayer.math.angle import degrees_point3, normalize_degrees_360
from cvlayer.math.norm import l2_norm
from cvlayer.typing import PointT, PolygonT

DEFAULT_MAX_MISSING_COUNT: Final[int] = 10
DEFAULT_MAX_STABLE_COUNT: Final[int] = 10
DEFAULT_STABLE_POINT_DELTA: Final[float] = 10.0


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


def normalize_degrees_180(angle: float) -> float:
    normalize_360 = normalize_degrees_360(angle)
    assert 0 <= normalize_360 < 360
    result = normalize_360 if normalize_360 <= 180 else normalize_360 - 360
    assert -180 < result <= 180
    return result


class RotateTracer:
    _first_polygon: Optional[PolygonT]
    _first_center: Optional[PointT]
    _first_point0: Optional[PointT]

    _current_polygon: Optional[PolygonT]
    _current_center: Optional[PointT]
    _current_point0: Optional[PointT]

    def __init__(
        self,
        max_missing_count=DEFAULT_MAX_MISSING_COUNT,
        max_stable_count=DEFAULT_MAX_STABLE_COUNT,
        stable_point_delta=DEFAULT_STABLE_POINT_DELTA,
    ):
        self._max_missing_count = max_missing_count
        self._missing_count = 0

        self._max_stable_count = max_stable_count
        self._stable_point_delta = stable_point_delta
        self._stable_count = 0

        self._first_polygon = None
        self._first_center = None
        self._first_point0 = None

        self._current_polygon = None
        self._current_center = None
        self._current_point0 = None
        self._current_rotate = 0.0

    @property
    def has_first_polygon(self) -> bool:
        return self._first_polygon is not None

    @property
    def is_stable_first(self) -> bool:
        return self._max_stable_count <= self._stable_count

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
    def max_stable_count(self) -> int:
        return self._max_stable_count

    @max_stable_count.setter
    def max_stable_count(self, value: int) -> None:
        self._max_stable_count = value

    @property
    def stable_count(self) -> int:
        return self._stable_count

    @property
    def current_center(self) -> Optional[Tuple[float, float]]:
        return self._current_center

    @property
    def current_point0(self) -> Optional[Tuple[float, float]]:
        return self._current_point0

    @property
    def current_rotate_delta(self) -> float:
        return normalize_degrees_180(self._current_rotate)

    @property
    def rotate_degrees(self) -> float:
        return self.current_rotate_delta

    def clear(self) -> None:
        self._missing_count = 0

        self._first_polygon = None
        self._first_center = None
        self._first_point0 = None

        self._current_polygon = None
        self._current_center = None
        self._current_point0 = None
        self._current_rotate = 0.0

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

        self._stable_count = 0

        self._first_polygon = deepcopy(polygon)
        self._first_center = deepcopy(center)
        self._first_point0 = (polygon[0][0], polygon[0][1])

        self._current_polygon = self._first_polygon
        self._current_center = self._first_center
        self._current_point0 = self._first_point0
        self._current_rotate = 0.0

    def on_trace_next_angle(self, polygon: PolygonT, center: PointT) -> None:
        assert polygon
        assert center
        assert isinstance(polygon, list)
        assert isinstance(center, tuple)

        assert self._first_point0 is not None
        assert self._first_center is not None
        assert self._current_point0 is not None

        next_point0 = find_nearest_point(self._current_point0, *polygon)
        next_center = deepcopy(center)
        next_degrees = calc_degrees(
            self._first_point0,
            self._first_center,
            next_point0,
            next_center,
        )

        self._current_polygon = polygon
        self._current_center = next_center
        self._current_point0 = next_point0
        self._current_rotate = next_degrees

        if self.is_stable_first:
            return

        p0w = l2_norm(
            x1=self._first_point0[0],
            y1=self._first_point0[1],
            x2=next_point0[0],
            y2=next_point0[1],
        )
        cw = l2_norm(
            x1=self._first_center[0],
            y1=self._first_center[1],
            x2=next_center[0],
            y2=next_center[1],
        )

        if self._stable_point_delta <= p0w or self._stable_point_delta <= cw:
            self._stable_count = 0
            self._first_polygon = deepcopy(polygon)
            self._first_center = next_center
            self._first_point0 = next_point0
        else:
            self._stable_count += 1

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

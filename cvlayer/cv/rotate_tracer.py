# -*- coding: utf-8 -*-

from typing import Final, Optional, Tuple

import cv2

from cvlayer.geometry.find_nearset_point import find_nearest_point
from cvlayer.math.angle import degrees_point3, normalize_signed_degrees_360
from cvlayer.typing import PointT, PolygonT

DEFAULT_MAX_MISSING_COUNT: Final[int] = 10
DEFAULT_MAX_STABLE_COUNT: Final[int] = 10
DEFAULT_STABLE_DEGREES_DELTA: Final[float] = 3.0
DEFAULT_MAX_ABNORMAL_COUNT: Final[int] = 10
DEFAULT_ABNORMAL_DEGREES_DELTA: Final[float] = 20.0


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


def in_degrees(angle: float, pivot: float, delta: float) -> bool:
    assert 0 <= angle < 360
    assert 0 <= pivot < 360
    assert 0 <= delta
    compare_angle = angle + delta
    min_degrees = pivot - delta + delta
    max_degrees = pivot + delta + delta
    return min_degrees <= compare_angle <= max_degrees


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
        stable_degrees_delta=DEFAULT_STABLE_DEGREES_DELTA,
        max_abnormal_count=DEFAULT_MAX_ABNORMAL_COUNT,
        abnormal_degrees_delta=DEFAULT_ABNORMAL_DEGREES_DELTA,
    ):
        self.max_missing_count = max_missing_count
        self.max_stable_count = max_stable_count
        self.stable_degrees_delta = stable_degrees_delta
        self.max_abnormal_count = max_abnormal_count
        self.abnormal_degrees_delta = abnormal_degrees_delta

        self._missing_count = 0
        self._current_rotate = 0.0

        self._stable_count = 0
        self._stable_rotate = 0.0
        self._stable_current_degrees = 0.0

        self._abnormal_count = 0

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
        return self._missing_count >= self.max_missing_count

    @property
    def missing_count(self) -> int:
        return self._missing_count

    @property
    def current_rotate(self) -> float:
        return self._current_rotate

    @property
    def stable_count(self) -> int:
        return self._stable_count

    @property
    def stable_rotate(self) -> float:
        return self._stable_rotate

    @property
    def abnormal_count(self) -> int:
        return self._abnormal_count

    @property
    def current_center(self) -> Optional[Tuple[float, float]]:
        return self._current_center

    @property
    def current_point0(self) -> Optional[Tuple[float, float]]:
        return self._current_point0

    @property
    def rotate_degrees(self) -> float:
        return normalize_signed_degrees_180(self._current_rotate - self._stable_rotate)

    def clear(self) -> None:
        self._missing_count = 0
        self._current_rotate = 0.0

        self._stable_count = 0
        self._stable_rotate = 0.0
        self._stable_current_degrees = 0.0

        self._abnormal_count = 0

        self._first_polygon = None
        self._first_center = None
        self._first_point0 = None

        self._current_polygon = None
        self._current_center = None
        self._current_point0 = None

    def do_missing(self) -> None:
        if not self.has_first_polygon:
            return

        self._missing_count += 1

        if self.overflow_missing_count:
            self.clear()

    def do_trace_first_angle(self, polygon: PolygonT, center: PointT) -> None:
        assert polygon
        assert center
        assert isinstance(polygon, list)
        assert isinstance(center, tuple)

        self._missing_count = 0
        self._current_rotate = 0.0
        self._stable_count = 0
        self._stable_rotate = 0.0
        self._stable_current_degrees = 0.0
        self._abnormal_count = 0

        self._first_polygon = polygon
        self._first_center = center
        self._first_point0 = (polygon[0][0], polygon[0][1])

        self._current_polygon = self._first_polygon
        self._current_center = self._first_center
        self._current_point0 = self._first_point0

    def do_trace_next_angle(self, polygon: PolygonT, center: PointT) -> None:
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

        assert 0 <= next_degrees < 360
        assert 0 <= self._current_rotate < 360
        assert 0 <= self.abnormal_degrees_delta
        if not in_degrees(
            angle=next_degrees,
            pivot=self._current_rotate,
            delta=self.abnormal_degrees_delta,
        ):
            self._abnormal_count += 1
            if self._abnormal_count < self.max_abnormal_count:
                return

        self._abnormal_count = 0
        self._current_polygon = polygon
        self._current_center = next_center
        self._current_point0 = next_point0
        self._current_rotate = next_degrees

        self._trace_stable_degrees(next_degrees)

    def _trace_stable_degrees(self, next_degrees: float) -> None:
        if self.max_stable_count <= self._stable_count:
            return

        assert 0 <= next_degrees < 360
        assert 0 <= self._stable_current_degrees < 360
        assert 0 <= self.stable_degrees_delta
        if in_degrees(
            angle=next_degrees,
            pivot=self._stable_current_degrees,
            delta=self.stable_degrees_delta,
        ):
            self._stable_count += 1
            next_stable_degrees = (self._stable_current_degrees + next_degrees) / 2.0
            self._stable_current_degrees = next_stable_degrees
            if self.max_stable_count <= self._stable_count:
                self._stable_rotate = self._stable_current_degrees
        else:
            self._stable_count = 0
            self._stable_current_degrees = next_degrees

    def do_trace(self, polygon: PolygonT, center: PointT) -> None:
        self._missing_count = 0

        if self.has_first_polygon:
            self.do_trace_next_angle(polygon, center)
        else:
            self.do_trace_first_angle(polygon, center)

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
    def cvl_create_rotate_tracer(
        max_missing_count=DEFAULT_MAX_MISSING_COUNT,
        max_stable_count=DEFAULT_MAX_STABLE_COUNT,
        stable_degrees_delta=DEFAULT_STABLE_DEGREES_DELTA,
        max_abnormal_count=DEFAULT_MAX_ABNORMAL_COUNT,
        abnormal_degrees_delta=DEFAULT_ABNORMAL_DEGREES_DELTA,
    ):
        return RotateTracer(
            max_missing_count=max_missing_count,
            max_stable_count=max_stable_count,
            stable_degrees_delta=stable_degrees_delta,
            max_abnormal_count=max_abnormal_count,
            abnormal_degrees_delta=abnormal_degrees_delta,
        )

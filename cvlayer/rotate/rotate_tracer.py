# -*- coding: utf-8 -*-

from collections import deque
from typing import Deque, Optional

from cvlayer.cv.types.shape import PointT
from cvlayer.rotate.protractor import Protractor


class HistoryItem:
    __slots__ = ("center", "point", "signed_angle")

    def __init__(self, center: PointT, point: PointT, signed_angle: float):
        self.center = center
        self.point = point
        self.signed_angle = signed_angle


class RotateTracer:
    _history: Deque[HistoryItem]
    _initial: Optional[HistoryItem]

    def __init__(self, history: Optional[int] = None):
        self._protractor = Protractor()
        self._history = deque(maxlen=history)
        self._initial = None
        self._accumulated_angle = 0.0
        self._current_angle = 0.0

    @property
    def protractor(self):
        return self._protractor

    @property
    def initial(self):
        return self._initial

    @property
    def angle(self):
        return self._current_angle

    @property
    def history(self):
        return self._history

    @property
    def max(self):
        return self._history.maxlen

    @property
    def latest(self):
        return self._history[-1]

    @property
    def empty(self):
        return len(self._history) == 0

    def __len__(self):
        return self._history.__len__()

    def clear(self) -> None:
        self._protractor.clear()
        self._history.clear()
        self._initial = None
        self._accumulated_angle = 0.0
        self._current_angle = 0.0

    def push(self, center: PointT, point: PointT) -> float:
        if self._initial is None:
            self._initial = HistoryItem(center, point, 0.0)

        if not self._protractor.has_first:
            self._protractor.set_first(center, point)

        self._protractor.set_second(center, point)

        if self._history.maxlen and self._history.maxlen <= len(self._history):
            self._history.popleft()

        partial_angle = self._protractor.signed_angle
        assert -180 < partial_angle <= 180

        self._current_angle = self._accumulated_angle + partial_angle
        self._history.append(HistoryItem(center, point, self._current_angle))

        # [IMPORTANT]
        # Beyond 180 degrees it inverts to a negative angle,
        # so the 'first point' updates every 90 degrees.
        if partial_angle <= -90 or 90 <= partial_angle:
            self._accumulated_angle = self._current_angle
            self._protractor.set_first(center, point)

        return self._current_angle

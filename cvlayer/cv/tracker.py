# -*- coding: utf-8 -*-

from typing import Tuple

import cv2
from numpy.typing import NDArray

from cvlayer.cv.tracking import TrackingMethod, tracker_create_method_name
from cvlayer.types import RectInt


class Tracker:
    def __init__(self, method: TrackingMethod, *args, **kwargs):
        create_tracker_name = tracker_create_method_name(method)
        if not hasattr(cv2, create_tracker_name):
            raise NotImplementedError

        create_tracker = getattr(cv2, create_tracker_name)
        self._tracker = create_tracker(*args, **kwargs)

        assert hasattr(self._tracker, "init")
        assert hasattr(self._tracker, "update")

        self._method = method
        self._initialized = False
        self._updated = False
        self._bounding_box = (0, 0, 0, 0)

    @property
    def method(self) -> TrackingMethod:
        return self._method

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def updated(self) -> bool:
        return self._updated

    @property
    def bounding_box(self) -> RectInt:
        return self._bounding_box

    def init(self, frame: NDArray, roi: RectInt) -> bool:
        self._initialized = self._tracker.init(frame, roi)
        return self._initialized

    def update(self, frame: NDArray) -> Tuple[bool, RectInt]:
        if self._initialized is None:
            raise RuntimeError("Tracker not initialized")

        self._updated, self._bounding_box = self._tracker.update(frame)
        return self._updated, self._bounding_box

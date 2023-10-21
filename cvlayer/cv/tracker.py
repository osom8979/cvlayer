# -*- coding: utf-8 -*-

from typing import Tuple

import cv2
from numpy.typing import NDArray

from cvlayer.cv.tracking import TrackingMethod, tracker_create_method_name
from cvlayer.typing import RectI


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
    def bounding_box(self) -> RectI:
        return self._bounding_box

    def init(self, frame: NDArray, roi: RectI) -> bool:
        self._initialized = self._tracker.init(frame, roi)
        return self._initialized

    def update(self, frame: NDArray) -> Tuple[bool, RectI]:
        if self._initialized is None:
            raise RuntimeError("Tracker not initialized")

        self._updated, self._bounding_box = self._tracker.update(frame)
        return self._updated, self._bounding_box


class CvlTracker:
    @staticmethod
    def cvl_create_tracker_dasiamrpn(*args, **kwargs):
        return Tracker(TrackingMethod.DaSiamRPN, *args, **kwargs)

    @staticmethod
    def cvl_create_tracker_boosting(*args, **kwargs):
        return Tracker(TrackingMethod.Boosting, *args, **kwargs)

    @staticmethod
    def cvl_create_tracker_mil(*args, **kwargs):
        return Tracker(TrackingMethod.MIL, *args, **kwargs)

    @staticmethod
    def cvl_create_tracker_kcf(*args, **kwargs):
        return Tracker(TrackingMethod.KCF, *args, **kwargs)

    @staticmethod
    def cvl_create_tracker_tld(*args, **kwargs):
        return Tracker(TrackingMethod.TLD, *args, **kwargs)

    @staticmethod
    def cvl_create_tracker_medianflow(*args, **kwargs):
        return Tracker(TrackingMethod.MedianFlow, *args, **kwargs)

    @staticmethod
    def cvl_create_tracker_goturn(*args, **kwargs):
        return Tracker(TrackingMethod.GOTURN, *args, **kwargs)

    @staticmethod
    def cvl_create_tracker_csrt(*args, **kwargs):
        return Tracker(TrackingMethod.CSRT, *args, **kwargs)

    @staticmethod
    def cvl_create_tracker_mosse(*args, **kwargs):
        return Tracker(TrackingMethod.MOSSE, *args, **kwargs)

# -*- coding: utf-8 -*-

from enum import Enum, auto, unique
from functools import lru_cache
from typing import Final, List

import cv2


@unique
class TrackingMethod(Enum):
    DaSiamRPN = auto()
    Boosting = auto()
    MIL = auto()
    KCF = auto()
    TLD = auto()
    MedianFlow = auto()
    GOTURN = auto()
    CSRT = auto()
    MOSSE = auto()


DEFAULT_SUPPORTED_TRACKER_ORDER = (
    TrackingMethod.MIL,
    TrackingMethod.KCF,
    TrackingMethod.TLD,
    TrackingMethod.Boosting,
    TrackingMethod.MedianFlow,
    TrackingMethod.GOTURN,  # Deep learning based
    TrackingMethod.DaSiamRPN,  # Deep learning based
)

TRACKER_CREATE_METHOD_PREFIX: Final[str] = "Tracker"
TRACKER_CREATE_METHOD_SUFFIX: Final[str] = "_create"


def tracker_create_method_name(tracker: TrackingMethod) -> str:
    return f"{TRACKER_CREATE_METHOD_PREFIX}{tracker.name}{TRACKER_CREATE_METHOD_SUFFIX}"


def supported_tracker(tracker: TrackingMethod) -> bool:
    create_method_name = tracker_create_method_name(tracker)
    return hasattr(cv2, create_method_name)


@lru_cache
def find_supported_tracker_names() -> List[str]:
    begin = len(TRACKER_CREATE_METHOD_PREFIX)
    end = -1 * len(TRACKER_CREATE_METHOD_SUFFIX)

    result = list()
    for attr in dir(cv2):
        if not attr.startswith(TRACKER_CREATE_METHOD_PREFIX):
            continue
        if not attr.endswith(TRACKER_CREATE_METHOD_SUFFIX):
            continue
        result.append(attr[begin:end])
    return result


@lru_cache
def supported_trackers() -> List[TrackingMethod]:
    return [m for m in TrackingMethod if supported_tracker(m)]


@lru_cache
def default_supported_tracker() -> TrackingMethod:
    trackers = supported_trackers()
    if not trackers:
        raise NotImplementedError

    for test_tracker in DEFAULT_SUPPORTED_TRACKER_ORDER:
        if test_tracker in trackers:
            return test_tracker
    return trackers[0]


if __name__ == "__main__":
    print(f"Find supported tracker names: {find_supported_tracker_names()}")
    print(f"Supported trackers: {supported_trackers()}")
    print(f"Default tracker: {default_supported_tracker()}")

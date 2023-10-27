# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Union

import cv2

DIST_LABEL_CCOMP = cv2.DIST_LABEL_CCOMP
DIST_LABEL_PIXEL = cv2.DIST_LABEL_PIXEL


@unique
class DistanceTransformLabel(Enum):
    """DistanceTransform algorithm flags"""

    CCOMP = DIST_LABEL_CCOMP
    """
    Each connected component of zeros in src (as well as all the non-zero pixels
    closest to the connected component) will be assigned the same label
    """

    PIXEL = DIST_LABEL_PIXEL
    """Each zero pixel (and all the non-zero pixels closest to it) gets its own label"""


DistanceTransformLabelLike = Union[DistanceTransformLabel, str, int]

_CCOMP = DistanceTransformLabel.CCOMP

DEFAULT_DISTANCE_TRANSFORM_LABEL: Final[DistanceTransformLabel] = _CCOMP
DISTANCE_TRANSFORM_LABEL_MAP: Final[Dict[str, int]] = {
    # DistanceType enum names
    "CCOMP": DIST_LABEL_CCOMP,
    "PIXEL": DIST_LABEL_PIXEL,
    # cv2 symbol suffix names
    "LABEL_CCOMP": DIST_LABEL_CCOMP,
    "LABEL_PIXEL": DIST_LABEL_PIXEL,
    # cv2 symbol full names
    "DIST_LABEL_CCOMP": DIST_LABEL_CCOMP,
    "DIST_LABEL_PIXEL": DIST_LABEL_PIXEL,
}


def normalize_distance_transform_label(label: DistanceTransformLabelLike) -> int:
    if isinstance(label, DistanceTransformLabel):
        return label.value
    elif isinstance(label, str):
        return DISTANCE_TRANSFORM_LABEL_MAP[label.upper()]
    elif isinstance(label, int):
        return label
    else:
        raise TypeError(f"Unsupported label type: {type(label).__name__}")

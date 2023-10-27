# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

import cv2

DIST_LABEL_CCOMP: Final[int] = cv2.DIST_LABEL_CCOMP
DIST_LABEL_PIXEL: Final[int] = cv2.DIST_LABEL_PIXEL


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

DEFAULT_DISTANCE_TRANSFORM_LABEL: Final[DistanceTransformLabelLike] = DIST_LABEL_CCOMP
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


def normalize_distance_transform_label(
    label: Optional[DistanceTransformLabelLike],
) -> int:
    if label is None:
        assert isinstance(DEFAULT_DISTANCE_TRANSFORM_LABEL, int)
        return DEFAULT_DISTANCE_TRANSFORM_LABEL

    if isinstance(label, DistanceTransformLabel):
        return label.value
    elif isinstance(label, str):
        return DISTANCE_TRANSFORM_LABEL_MAP[label.upper()]
    elif isinstance(label, int):
        return label
    else:
        raise TypeError(f"Unsupported label type: {type(label).__name__}")

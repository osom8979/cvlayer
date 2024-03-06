# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

import cv2

DRAW_MATCHES_FLAGS_DEFAULT = cv2.DRAW_MATCHES_FLAGS_DEFAULT
DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS = (
    cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
)
DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS


@unique
class DrawMatches(Enum):
    DEFAULT = DRAW_MATCHES_FLAGS_DEFAULT
    DRAW_OVER_OUTIMG = DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
    NOT_DRAW_SINGLE_POINTS = DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    DRAW_RICH_KEYPOINTS = DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS


DrawMatchesLike = Union[DrawMatches, str, int]

DEFAULT_DRAW_MATCHES: Final[DrawMatchesLike] = DRAW_MATCHES_FLAGS_DEFAULT
DRAW_MATCHES_MAP: Final[Dict[str, int]] = {
    # cv2 symbol full names
    "DRAW_MATCHES_FLAGS_DEFAULT": DRAW_MATCHES_FLAGS_DEFAULT,
    "DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG": DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
    "DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS": (
        DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    ),
    "DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS": DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    # DrawMatchesLike enum names
    "DEFAULT": DRAW_MATCHES_FLAGS_DEFAULT,
    "DRAW_OVER_OUTIMG": DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
    "NOT_DRAW_SINGLE_POINTS": DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
    "DRAW_RICH_KEYPOINTS": DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
}


def normalize_draw_matches(draw_matches: Optional[DrawMatchesLike]) -> int:
    if draw_matches is None:
        assert isinstance(DEFAULT_DRAW_MATCHES, int)
        return DEFAULT_DRAW_MATCHES

    if isinstance(draw_matches, DrawMatches):
        return draw_matches.value
    elif isinstance(draw_matches, str):
        return DRAW_MATCHES_MAP[draw_matches.upper()]
    elif isinstance(draw_matches, int):
        return draw_matches
    else:
        raise TypeError(f"Unsupported draw matches: {type(draw_matches).__name__}")

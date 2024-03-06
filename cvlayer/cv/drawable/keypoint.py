# -*- coding: utf-8 -*-

from typing import Sequence

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.draw_matches import DEFAULT_DRAW_MATCHES, normalize_draw_matches


def draw_keypoints(
    image: NDArray,
    keypoints: Sequence[cv2.KeyPoint],
    flags=DEFAULT_DRAW_MATCHES,
) -> NDArray:
    _flags = normalize_draw_matches(flags)
    return cv2.drawKeypoints(
        image,
        keypoints,
        image.copy(),
        flags=_flags,
    )


class CvlDrawableKeyPoints:
    @staticmethod
    def cvl_draw_keypoints(
        image: NDArray,
        keypoints: Sequence[cv2.KeyPoint],
        flags=DEFAULT_DRAW_MATCHES,
    ):
        return draw_keypoints(image, keypoints, flags)

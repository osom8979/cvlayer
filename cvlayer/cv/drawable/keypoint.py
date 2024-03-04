# -*- coding: utf-8 -*-

from typing import Sequence

import cv2
from numpy.typing import NDArray


def draw_keypoints(
    image: NDArray,
    keypoints: Sequence[cv2.KeyPoint],
) -> NDArray:
    return cv2.drawKeypoints(image, keypoints, image.copy())


class CvlDrawableKeyPoints:
    @staticmethod
    def cvl_draw_keypoints(
        image: NDArray,
        keypoints: Sequence[cv2.KeyPoint],
    ):
        return draw_keypoints(image, keypoints)

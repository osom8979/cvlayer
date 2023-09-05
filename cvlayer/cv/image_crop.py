# -*- coding: utf-8 -*-

from numpy.typing import NDArray

from cvlayer.typing import RectInt


def crop(src: NDArray, roi: RectInt) -> NDArray:
    x1, y1, x2, y2 = roi
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    area = width * height
    if area == 0:
        raise ValueError("ROIs with size 0 cannot be cropped")
    return src[y1:y2, x1:x2]


class CvlImageCrop:
    @staticmethod
    def cvl_crop(src: NDArray, roi: RectInt):
        return crop(src, roi)

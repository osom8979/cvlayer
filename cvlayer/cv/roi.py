# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.typing import NumberT, RectT


def normalize_coord(
    value: NumberT,
    minval: Optional[NumberT] = 0,
    maxval: Optional[NumberT] = None,
) -> NumberT:
    if minval is not None and value < minval:
        return minval
    if maxval is not None and value > maxval:
        return maxval
    return value


def normalize_roi(
    roi: RectT,
    width: Optional[NumberT] = None,
    height: Optional[NumberT] = None,
) -> RectT:
    x1, y1, x2, y2 = roi

    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)

    return (
        normalize_coord(left, 0, width),
        normalize_coord(top, 0, height),
        normalize_coord(right, 0, width),
        normalize_coord(bottom, 0, height),
    )


def normalize_image_roi(image: NDArray, roi: RectT) -> RectT:
    return normalize_roi(roi, image.shape[1], image.shape[0])


class CvlRoi:
    @staticmethod
    def cvl_normalize_coord(
        value: NumberT,
        minval: Optional[NumberT] = 0,
        maxval: Optional[NumberT] = None,
    ):
        return normalize_coord(value, minval, maxval)

    @staticmethod
    def cvl_normalize_roi(
        roi: RectT,
        width: Optional[NumberT] = None,
        height: Optional[NumberT] = None,
    ):
        return normalize_roi(roi, width, height)

    @staticmethod
    def cvl_normalize_image_roi(image: NDArray, roi: RectT) -> RectT:
        return normalize_image_roi(image, roi)

# -*- coding: utf-8 -*-

from typing import Optional, Sequence

from numpy import full
from numpy.typing import NDArray

from cvlayer.typing import RectI


def image_crop(src: NDArray, roi: RectI, copy=False) -> NDArray:
    x1, y1, x2, y2 = roi
    width = x2 - x1
    height = y2 - y1
    area = width * height
    if area == 0:
        raise ValueError("ROIs with size 0 cannot be cropped")

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    src_height, src_width = src.shape[0], src.shape[1]
    if left < 0 or src_width < left:
        raise IndexError(f"Invalid roi: {roi}")
    if right < 0 or src_width < right:
        raise IndexError(f"Invalid roi: {roi}")
    if top < 0 or src_height < top:
        raise IndexError(f"Invalid roi: {roi}")
    if bottom < 0 or src_height < bottom:
        raise IndexError(f"Invalid roi: {roi}")

    cropped = src[top:bottom, left:right]
    if copy:
        return cropped.copy()
    else:
        return cropped


def image_crop_adjusted(
    src: NDArray,
    roi: RectI,
    copy=False,
) -> NDArray:
    x1, y1, x2, y2 = roi
    src_height, src_width = src.shape[0], src.shape[1]
    left = max(min(x1, x2), 0)
    right = min(max(x1, x2), src_width)
    top = max(min(y1, y2), 0)
    bottom = min(max(y1, y2), src_height)

    width = right - left
    height = bottom - top
    area = width * height
    if area == 0:
        raise ValueError("ROIs with size 0 cannot be cropped")

    cropped = src[top:bottom, left:right]
    if copy:
        return cropped.copy()
    else:
        return cropped


def image_crop_extended(
    src: NDArray,
    roi: RectI,
    fill: Optional[Sequence[int]] = None,
) -> NDArray:
    x1, y1, x2, y2 = roi
    width = x2 - x1
    height = y2 - y1
    area = width * height
    if area == 0:
        raise ValueError("ROIs with size 0 cannot be cropped")

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    src_height, src_width = src.shape[0], src.shape[1]
    crop_left = max(left, 0)
    crop_right = min(right, src_width)
    crop_top = max(top, 0)
    crop_bottom = min(bottom, src_height)

    left_over = abs(left - crop_left)
    right_over = abs(right - crop_right)
    top_over = abs(top - crop_top)
    bottom_over = abs(bottom - crop_bottom)

    cropped = src[crop_top:crop_bottom, crop_left:crop_right]
    extend_height = cropped.shape[0] + top_over + bottom_over
    extend_width = cropped.shape[1] + left_over + right_over
    extend_shape = [extend_height, extend_width]
    if len(src.shape) >= 3:
        extend_shape += cropped.shape[2:]

    canvas = full(extend_shape, fill if fill is not None else 0, dtype=cropped.dtype)
    canvas_y1 = top_over
    canvas_y2 = top_over + cropped.shape[0]
    canvas_x1 = left_over
    canvas_x2 = left_over + cropped.shape[1]
    canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = cropped
    return canvas


class CvlImageCrop:
    @staticmethod
    def cvl_image_crop(src: NDArray, roi: RectI, copy=False):
        return image_crop(src, roi, copy)

    @staticmethod
    def cvl_image_crop_adjusted(
        src: NDArray,
        roi: RectI,
        copy=False,
    ) -> NDArray:
        return image_crop_adjusted(src, roi, copy)

    @staticmethod
    def cvl_image_crop_extended(
        src: NDArray,
        roi: RectI,
        fill: Optional[Sequence[int]] = None,
    ) -> NDArray:
        return image_crop_extended(src, roi, fill)

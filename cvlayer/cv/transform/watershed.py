# -*- coding: utf-8 -*-

import cv2
from numpy import int8, int32, uint8, uint32
from numpy.typing import NDArray


def watershed(image: NDArray, markers: NDArray) -> NDArray:
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"The image argument must be 3-channels: {image.shape}")
    if image.dtype not in (int8, uint8):
        raise TypeError(f"The image argument must be 8-bits: {image.dtype}")

    if len(markers.shape) != 2:
        raise ValueError(f"The markers arg must be single-channels: {markers.shape}")
    if markers.dtype not in (int32, uint32):
        raise TypeError(f"The markers arg must be 32-bits: {markers.dtype}")

    image_height, image_width = image.shape[0:2]
    markers_height, markers_width = markers.shape

    if image_height != markers_height or image_width != markers_width:
        raise ValueError("The markers must be the same size as the image")

    return cv2.watershed(image, markers)


class CvlTransformWatershed:
    @staticmethod
    def cvl_watershed(image: NDArray, markers: NDArray):
        return watershed(image, markers)

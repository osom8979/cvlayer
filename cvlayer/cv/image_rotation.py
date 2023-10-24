# -*- coding: utf-8 -*-

from typing import Optional, Sequence

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.border import BorderType
from cvlayer.cv.types.interpolation import (
    DEFAULT_INTERPOLATION,
    normalize_interpolation,
)
from cvlayer.typing import Number


def image_rotation(
    src: NDArray,
    degrees: float,
    center_x: Optional[Number] = None,
    center_y: Optional[Number] = None,
    scale=1.0,
    result_width: Optional[Number] = None,
    result_height: Optional[Number] = None,
    interpolation=DEFAULT_INTERPOLATION,
    border_mode=BorderType.CONSTANT,
    border_value: Optional[Sequence[float]] = None,
) -> NDArray:
    cx = center_x if center_x else (src.shape[1] // 2)
    cy = center_y if center_y else (src.shape[0] // 2)
    center = int(cx), int(cy)
    matrix = cv2.getRotationMatrix2D(center=center, angle=degrees, scale=scale)
    width = result_width if result_width else src.shape[1]
    height = result_height if result_height else src.shape[0]
    dsize = int(width), int(height)
    flags = normalize_interpolation(interpolation)
    mode = border_mode.value
    value = border_value if border_value is not None else list()
    return cv2.warpAffine(
        src=src,
        M=matrix,
        dsize=dsize,
        dst=None,
        flags=flags,
        borderMode=mode,
        borderValue=value,
    )


class CvlImageRotation:
    @staticmethod
    def cvl_image_rotation(
        src: NDArray,
        degrees: float,
        center_x: Optional[Number] = None,
        center_y: Optional[Number] = None,
        scale=1.0,
        result_width: Optional[int] = None,
        result_height: Optional[int] = None,
        interpolation=DEFAULT_INTERPOLATION,
        border_mode=BorderType.CONSTANT,
        border_value: Optional[Sequence[float]] = None,
    ):
        return image_rotation(
            src=src,
            center_x=center_x,
            center_y=center_y,
            degrees=degrees,
            scale=scale,
            result_width=result_width,
            result_height=result_height,
            interpolation=interpolation,
            border_mode=border_mode,
            border_value=border_value,
        )

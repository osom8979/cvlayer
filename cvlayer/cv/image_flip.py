# -*- coding: utf-8 -*-

from enum import Enum, unique

import cv2
from numpy.typing import NDArray


@unique
class FlipCode(Enum):
    X_AXIS = 0
    Y_AXIS = 1
    XY_AXIS = -1


def image_flip(src: NDArray, flip_code: FlipCode) -> NDArray:
    return cv2.flip(src, flip_code.value)


def image_flip_x_axis(src: NDArray) -> NDArray:
    return image_flip(src, FlipCode.X_AXIS)


def image_flip_y_axis(src: NDArray) -> NDArray:
    return image_flip(src, FlipCode.Y_AXIS)


def image_flip_xy_axis(src: NDArray) -> NDArray:
    return image_flip(src, FlipCode.XY_AXIS)


class CvlImageFlip:
    @staticmethod
    def cvl_image_flip(src: NDArray, flip_code: FlipCode) -> NDArray:
        return image_flip(src, flip_code)

    @staticmethod
    def cvl_image_flip_x_axis(src: NDArray) -> NDArray:
        return image_flip_x_axis(src)

    @staticmethod
    def cvl_image_flip_y_axis(src: NDArray) -> NDArray:
        return image_flip_y_axis(src)

    @staticmethod
    def cvl_image_flip_xy_axis(src: NDArray) -> NDArray:
        return image_flip_xy_axis(src)

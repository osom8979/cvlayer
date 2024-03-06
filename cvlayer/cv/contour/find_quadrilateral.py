# -*- coding: utf-8 -*-

import cv2
from numpy.typing import NDArray

from cvlayer.cv.contour.analysis import RotatedRect, min_area_rect


def find_quadrilateral_with_rotated_rect(
    contour: NDArray,
    rotated_rect: RotatedRect,
):
    center = rotated_rect.center
    size = rotated_rect.size
    rotation = rotated_rect.rotation


def find_quadrilateral(contour: NDArray):
    return find_quadrilateral_with_rotated_rect(contour, min_area_rect(contour))


class CvlFindQuadrilateral:
    @staticmethod
    def cvl_find_quadrilateral_with_rotated_rect(
        contour: NDArray,
        rotated_rect: RotatedRect,
    ):
        return find_quadrilateral_with_rotated_rect(contour, rotated_rect)

    @staticmethod
    def cvl_find_quadrilateral(contour: NDArray):
        return find_quadrilateral(contour)

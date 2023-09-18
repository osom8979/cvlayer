# -*- coding: utf-8 -*-

from numpy import hstack, vstack
from numpy.typing import NDArray


def horizontal_stack(*images: NDArray) -> NDArray:
    return hstack(images)


def vertical_stack(*images: NDArray) -> NDArray:
    return vstack(images)


class CvlStack:
    @staticmethod
    def cvl_horizontal_stack(*images: NDArray):
        return horizontal_stack(*images)

    @staticmethod
    def cvl_vertical_stack(*images: NDArray):
        return vertical_stack(*images)

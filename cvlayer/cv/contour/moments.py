# -*- coding: utf-8 -*-

from dataclasses import dataclass

import cv2
from numpy.typing import NDArray


@dataclass
class Moments:
    # Spatial Moments
    m00: float
    m10: float
    m01: float
    m20: float
    m11: float
    m02: float
    m30: float
    m21: float
    m12: float
    m03: float

    # Central Moments
    mu20: float
    mu11: float
    mu02: float
    mu30: float
    mu21: float
    mu12: float
    mu03: float

    # Central Normalized Moments
    nu20: float
    nu11: float
    nu02: float
    nu30: float
    nu21: float
    nu12: float
    nu03: float

    @property
    def center_x(self):
        return self.m10 / self.m00

    @property
    def center_y(self):
        return self.m01 / self.m00

    @property
    def center(self):
        return self.center_x, self.center_y


def moments(contour: NDArray, binary_image=False) -> Moments:
    """
    Only applicable to contour moments calculations from Python bindings.

    Note that the numpy type for the input array should be either
    `np.int32` or `np.float32`.
    """
    m = cv2.moments(contour, binaryImage=binary_image)

    return Moments(
        m00=m["m00"],
        m10=m["m10"],
        m01=m["m01"],
        m20=m["m20"],
        m11=m["m11"],
        m02=m["m02"],
        m30=m["m30"],
        m21=m["m21"],
        m12=m["m12"],
        m03=m["m03"],
        mu20=m["mu20"],
        mu11=m["mu11"],
        mu02=m["mu02"],
        mu30=m["mu30"],
        mu21=m["mu21"],
        mu12=m["mu12"],
        mu03=m["mu03"],
        nu20=m["nu20"],
        nu11=m["nu11"],
        nu02=m["nu02"],
        nu30=m["nu30"],
        nu21=m["nu21"],
        nu12=m["nu12"],
        nu03=m["nu03"],
    )


class CvlContourMoments:
    @staticmethod
    def cvl_moments(contour: NDArray, binary_image=False):
        return moments(contour, binary_image)

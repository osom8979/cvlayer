# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final, Optional

import cv2
from numpy import array, float32
from numpy.typing import NDArray

from cvlayer.typing import PointFloat, RectFloat, Scalar, SizeInt


@unique
class MatrixDecomposition(Enum):
    LU = cv2.DECOMP_LU
    """Gaussian elimination with the optimal pivot element chosen"""

    SVD = cv2.DECOMP_SVD
    """
    singular value decomposition (SVD) method;
    the system can be over-defined and/or the matrix src1 can be singular
    """

    EIG = cv2.DECOMP_EIG
    """eigenvalue decomposition;
    the matrix src1 must be symmetrical
    """

    CHOLESKY = cv2.DECOMP_CHOLESKY
    """Cholesky LLT factorization;
    the matrix src1 must be symmetrical and positively defined
    """

    QR = cv2.DECOMP_QR
    """
    QR factorization;
    the system can be over-defined and/or the matrix src1 can be singular
    """

    NORMAL = cv2.DECOMP_NORMAL
    """
    while all the previous flags are mutually exclusive,
    this flag can be used together with any of the previous;
    """


def get_perspective_transform(
    src: NDArray,
    dest: NDArray,
    solve_method=MatrixDecomposition.LU,
) -> NDArray:
    return cv2.getPerspectiveTransform(src, dest, solve_method.value)


def get_perspective_transform_with_quadrilateral(
    left_top: PointFloat,
    left_bottom: PointFloat,
    right_top: PointFloat,
    right_bottom: PointFloat,
    destination_roi: RectFloat,
    solve_method=MatrixDecomposition.LU,
) -> NDArray:
    points = left_top, left_bottom, right_top, right_bottom
    src_coordinates = array(points, dtype=float32)

    x1, y1, x2, y2 = destination_roi
    dst_coordinates = array(((x1, y1), (x1, y2), (x2, y1), (x2, y2)), dtype=float32)

    return get_perspective_transform(src_coordinates, dst_coordinates, solve_method)


INTER_LINEAR: Final[int] = cv2.INTER_LINEAR
INTER_NEAREST: Final[int] = cv2.INTER_NEAREST
WARP_INVERSE_MAP: Final[int] = cv2.WARP_INVERSE_MAP
"""
`WARP_INVERSE_MAP` that sets `M` as the inverse transformation (`dst -> src`)
"""


@unique
class WarpPerspectiveFlags(Enum):
    LINEAR = INTER_LINEAR
    NEAREST = INTER_NEAREST
    LINEAR_INVERSE = INTER_LINEAR + WARP_INVERSE_MAP
    NEAREST_INVERSE = INTER_NEAREST + WARP_INVERSE_MAP


@unique
class WarpPerspectiveBorderMode(Enum):
    CONSTANT = cv2.BORDER_CONSTANT
    REPLICATE = cv2.BORDER_REPLICATE


def warp_perspective(
    image: NDArray,
    matrix: NDArray,
    dsize: SizeInt,
    flags=WarpPerspectiveFlags.LINEAR,
    border_mode=WarpPerspectiveBorderMode.CONSTANT,
    border_value: Optional[Scalar] = None,
) -> NDArray:
    return cv2.warpPerspective(
        image,
        matrix,
        dsize,
        flags=flags.value,
        borderMode=border_mode.value,
        borderValue=border_value,
    )


class CvlPerspective:
    @staticmethod
    def cvl_get_perspective_transform(
        src: NDArray,
        dest: NDArray,
        solve_method=MatrixDecomposition.LU,
    ):
        return get_perspective_transform(src, dest, solve_method)

    @staticmethod
    def cvl_get_perspective_transform_with_quadrilateral(
        left_top: PointFloat,
        left_bottom: PointFloat,
        right_top: PointFloat,
        right_bottom: PointFloat,
        destination_roi: RectFloat,
        solve_method=MatrixDecomposition.LU,
    ):
        return get_perspective_transform_with_quadrilateral(
            left_top,
            left_bottom,
            right_top,
            right_bottom,
            destination_roi,
            solve_method,
        )

    @staticmethod
    def cvl_warp_perspective(
        image: NDArray,
        matrix: NDArray,
        dsize: SizeInt,
        flags=WarpPerspectiveFlags.LINEAR,
        border_mode=WarpPerspectiveBorderMode.CONSTANT,
        border_value: Optional[Scalar] = None,
    ):
        return warp_perspective(
            image,
            matrix,
            dsize,
            flags,
            border_mode,
            border_value,
        )

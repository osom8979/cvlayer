# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Any, Final, Optional

import cv2
from numpy import array, float32
from numpy.typing import NDArray

from cvlayer.typing import PerspectivePointsI, PointF, RectF, ScalarF, SizeN


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


def cast_perspective_int(points: Any) -> PerspectivePointsI:
    assert isinstance(points, (tuple, list))
    assert len(points) == 4

    assert isinstance(points[0], (tuple, list))
    assert len(points[0]) == 2
    assert isinstance(points[0][0], int)
    assert isinstance(points[0][1], int)

    assert isinstance(points[1], (tuple, list))
    assert len(points[1]) == 2
    assert isinstance(points[1][0], int)
    assert isinstance(points[1][1], int)

    assert isinstance(points[2], (tuple, list))
    assert len(points[2]) == 2
    assert isinstance(points[2][0], int)
    assert isinstance(points[2][1], int)

    assert isinstance(points[3], (tuple, list))
    assert len(points[3]) == 2
    assert isinstance(points[3][0], int)
    assert isinstance(points[3][1], int)

    return (
        (points[0][0], points[0][1]),
        (points[1][0], points[1][1]),
        (points[2][0], points[2][1]),
        (points[3][0], points[3][1]),
    )


def get_perspective_transform(
    src: NDArray,
    dest: NDArray,
    solve_method=MatrixDecomposition.LU,
) -> NDArray:
    return cv2.getPerspectiveTransform(src, dest, solve_method.value)


def get_perspective_transform_with_quadrilateral(
    left_top: PointF,
    left_bottom: PointF,
    right_top: PointF,
    right_bottom: PointF,
    destination_roi: RectF,
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
    dsize: SizeN,
    flags=WarpPerspectiveFlags.LINEAR,
    border_mode=WarpPerspectiveBorderMode.CONSTANT,
    border_value: Optional[ScalarF] = None,
) -> NDArray:
    return cv2.warpPerspective(
        image,
        matrix,
        dsize=(int(dsize[0]), int(dsize[1])),
        flags=flags.value,
        borderMode=border_mode.value,
        borderValue=border_value if border_value else tuple(),
    )


class CvlPerspective:
    @staticmethod
    def cvl_cast_perspective_int(points: Any) -> PerspectivePointsI:
        return cast_perspective_int(points)

    @staticmethod
    def cvl_get_perspective_transform(
        src: NDArray,
        dest: NDArray,
        solve_method=MatrixDecomposition.LU,
    ):
        return get_perspective_transform(src, dest, solve_method)

    @staticmethod
    def cvl_get_perspective_transform_with_quadrilateral(
        left_top: PointF,
        left_bottom: PointF,
        right_top: PointF,
        right_bottom: PointF,
        destination_roi: RectF,
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
        dsize: SizeN,
        flags=WarpPerspectiveFlags.LINEAR,
        border_mode=WarpPerspectiveBorderMode.CONSTANT,
        border_value: Optional[ScalarF] = None,
    ):
        return warp_perspective(
            image,
            matrix,
            dsize,
            flags,
            border_mode,
            border_value,
        )

# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.perspective import (
    MatrixDecomposition,
    WarpPerspectiveBorderMode,
    WarpPerspectiveFlags,
    get_perspective_transform,
    get_perspective_transform_with_quadrilateral,
    warp_perspective,
)
from cvlayer.types import PointFloat, RectFloat, Scalar, SizeInt


class CvlPerspective:
    @staticmethod
    def cvl_get_perspective_transform(
        src: NDArray,
        dest: NDArray,
        solve_method=MatrixDecomposition.LU,
    ) -> NDArray:
        return get_perspective_transform(src, dest, solve_method)

    @staticmethod
    def cvl_get_perspective_transform_with_quadrilateral(
        left_top: PointFloat,
        left_bottom: PointFloat,
        right_top: PointFloat,
        right_bottom: PointFloat,
        destination_roi: RectFloat,
        solve_method=MatrixDecomposition.LU,
    ) -> NDArray:
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
    ) -> NDArray:
        return warp_perspective(
            image,
            matrix,
            dsize,
            flags,
            border_mode,
            border_value,
        )

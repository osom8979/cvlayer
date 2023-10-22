# -*- coding: utf-8 -*-

from typing import Final, Optional, Sequence

import cv2
from numpy import int64
from numpy.typing import NDArray

from cvlayer.cv.contours import RotatedRect, box_points
from cvlayer.cv.drawable.defaults import (
    DEFAULT_COLOR,
    DEFAULT_LINE_TYPE,
    DEFAULT_THICKNESS,
)

DRAW_ALL_CONTOURS: Final[int] = -1


def draw_contour(
    image: NDArray,
    contour: NDArray,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line_type=DEFAULT_LINE_TYPE,
) -> None:
    cv2.drawContours(image, [contour], 0, color, thickness, line_type)


def draw_contours(
    image: NDArray,
    contours: Sequence[NDArray],
    contour_index=DRAW_ALL_CONTOURS,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line_type=DEFAULT_LINE_TYPE,
    hierarchy: Optional[NDArray] = None,
) -> None:
    cv2.drawContours(
        image,
        contours,
        contour_index,
        color,
        thickness,
        line_type,
        hierarchy,
    )


def draw_min_area_rect(
    image: NDArray,
    box: RotatedRect,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line_type=DEFAULT_LINE_TYPE,
) -> None:
    points = box_points(box)
    int64_points = int64(points)
    draw_contour(
        image,
        int64_points,  # type: ignore[arg-type]
        color,
        thickness,
        line_type,
    )


class CvlDrawableContours:
    @staticmethod
    def cvl_draw_contour(
        image: NDArray,
        contour: NDArray,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line_type=DEFAULT_LINE_TYPE,
    ):
        draw_contour(image, contour, color, thickness, line_type)

    @staticmethod
    def cvl_draw_contours(
        image: NDArray,
        contours: Sequence[NDArray],
        contour_index=DRAW_ALL_CONTOURS,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line_type=DEFAULT_LINE_TYPE,
        hierarchy: Optional[NDArray] = None,
    ):
        draw_contours(
            image,
            contours,
            contour_index,
            color,
            thickness,
            line_type,
            hierarchy,
        )

    @staticmethod
    def cvl_draw_min_box_area(
        image: NDArray,
        box: RotatedRect,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line_type=DEFAULT_LINE_TYPE,
    ):
        draw_min_area_rect(image, box, color, thickness, line_type)

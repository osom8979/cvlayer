# -*- coding: utf-8 -*-

from typing import Final, Optional, Sequence

import cv2
from numpy import int64
from numpy.typing import NDArray

from cvlayer.cv.contours import MinAreaRectResult, box_points
from cvlayer.cv.drawable import COLOR, LINE_TYPE, THICKNESS
from cvlayer.typing import Image

DRAW_ALL_CONTOURS: Final[int] = -1


def draw_contour(
    image: Image,
    contour: NDArray,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
) -> None:
    cv2.drawContours(image, [contour], 0, color, thickness, line_type)


def draw_contours(
    image: Image,
    contours: Sequence[NDArray],
    contour_index=DRAW_ALL_CONTOURS,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
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
    image: Image,
    box: MinAreaRectResult,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
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
        image: Image,
        contour: NDArray,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
    ):
        draw_contour(image, contour, color, thickness, line_type)

    @staticmethod
    def cvl_draw_contours(
        image: Image,
        contours: Sequence[NDArray],
        contour_index=DRAW_ALL_CONTOURS,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
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
        image: Image,
        box: MinAreaRectResult,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
    ):
        draw_min_area_rect(image, box, color, thickness, line_type)

# -*- coding: utf-8 -*-

from typing import Optional, Sequence

from numpy.typing import NDArray

from cvlayer.cv.drawable_contours import (
    COLOR,
    CONTOURS_ALL,
    LINE_TYPE,
    THICKNESS,
    MinAreaRectResult,
    draw_contour,
    draw_contours,
    draw_min_area_rect,
)
from cvlayer.types import Image


class CvlDrawableContours:
    @staticmethod
    def cvl_draw_contour(
        image: Image,
        contour: NDArray,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
    ) -> None:
        draw_contour(image, contour, color, thickness, line_type)

    @staticmethod
    def cvl_draw_contours(
        image: Image,
        contours: Sequence[NDArray],
        contour_index=CONTOURS_ALL,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
        hierarchy: Optional[NDArray] = None,
    ) -> None:
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
    ) -> None:
        draw_min_area_rect(image, box)

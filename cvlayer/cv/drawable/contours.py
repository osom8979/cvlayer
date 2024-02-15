# -*- coding: utf-8 -*-

from typing import Final, Optional, Sequence

import cv2
from numpy import int64
from numpy.typing import NDArray

from cvlayer.cv.contour.analysis import RotatedRect, box_points
from cvlayer.cv.drawable.defaults import (
    DEFAULT_COLOR,
    DEFAULT_LINE_TYPE,
    DEFAULT_THICKNESS,
)
from cvlayer.cv.types.color import normalize_color
from cvlayer.cv.types.line_type import normalize_line_type
from cvlayer.math.climit import INT_MAX
from cvlayer.typing import PointI

DRAW_ALL_CONTOURS: Final[int] = -1


def draw_contours(
    image: NDArray,
    contours: Sequence[NDArray],
    index=DRAW_ALL_CONTOURS,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    hierarchy: Optional[NDArray] = None,
    max_level=INT_MAX,
    offset: Optional[PointI] = None,
) -> NDArray:
    _color = normalize_color(color)
    _line = normalize_line_type(line)
    return cv2.drawContours(
        image,
        contours,
        index,
        _color,
        thickness,
        _line,
        hierarchy,
        max_level,
        offset,  # type: ignore[arg-type]
    )


def draw_contour(
    image: NDArray,
    contour: NDArray,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    offset: Optional[PointI] = None,
) -> NDArray:
    return draw_contours(
        image,
        [contour],
        0,
        color,
        thickness,
        line,
        offset=offset,
    )


def draw_min_area_rect(
    image: NDArray,
    box: RotatedRect,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    offset: Optional[PointI] = None,
) -> NDArray:
    points = box_points(box)
    int64_points = int64(points)
    return draw_contour(
        image,
        int64_points,  # type: ignore[arg-type]
        color,
        thickness,
        line,
        offset,
    )


class CvlDrawableContours:
    @staticmethod
    def cvl_draw_contours(
        image: NDArray,
        contours: Sequence[NDArray],
        contour_index=DRAW_ALL_CONTOURS,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        hierarchy: Optional[NDArray] = None,
        max_level=INT_MAX,
        offset: Optional[PointI] = None,
    ):
        return draw_contours(
            image,
            contours,
            contour_index,
            color,
            thickness,
            line,
            hierarchy,
            max_level,
            offset,
        )

    @staticmethod
    def cvl_draw_contour(
        image: NDArray,
        contour: NDArray,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        offset: Optional[PointI] = None,
    ):
        return draw_contour(image, contour, color, thickness, line, offset)

    @staticmethod
    def cvl_draw_min_box_area(
        image: NDArray,
        box: RotatedRect,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        offset: Optional[PointI] = None,
    ):
        return draw_min_area_rect(image, box, color, thickness, line, offset)

# -*- coding: utf-8 -*-

from typing import Final, Optional, Sequence

import cv2
from numpy import int64
from numpy.typing import NDArray

from cvlayer.cv.contours import MinAreaRectResult, box_points
from cvlayer.cv.drawable import COLOR, LINE_TYPE, THICKNESS
from cvlayer.types import Image

CONTOURS_ALL: Final[int] = -1


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
    contour_index=CONTOURS_ALL,
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


def draw_min_area_rect(image: Image, box: MinAreaRectResult) -> None:
    points = box_points(box)
    int64_points = int64(points)
    draw_contour(image, int64_points)  # type: ignore[arg-type]

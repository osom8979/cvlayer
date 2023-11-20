# -*- coding: utf-8 -*-

from typing import Final, Union

from numpy.typing import NDArray

from cvlayer.cv.drawable.circle import draw_circle_coord
from cvlayer.cv.drawable.line import draw_line
from cvlayer.cv.types.color import Color
from cvlayer.cv.types.line_type import LineType, LineTypeLike
from cvlayer.palette.basic import RED
from cvlayer.typing import Number, PointN

CROSSHAIR_POINT_RADIUS: Final[int] = 6
CROSSHAIR_POINT_THICKNESS: Final[int] = 1
CROSSHAIR_POINT_COLOR: Final[Union[Color, int, str]] = RED
CROSSHAIR_POINT_LINE_TYPE: Final[LineTypeLike] = LineType.AA
CROSSHAIR_POINT_SHIFT: Final[int] = 0
CROSSHAIR_POINT_PADDING: Final[int] = 2


def draw_crosshair_coord(
    image: NDArray,
    x: Number,
    y: Number,
    radius=CROSSHAIR_POINT_RADIUS,
    thickness=CROSSHAIR_POINT_THICKNESS,
    color=CROSSHAIR_POINT_COLOR,
    line=CROSSHAIR_POINT_LINE_TYPE,
    shift=CROSSHAIR_POINT_SHIFT,
    padding=CROSSHAIR_POINT_PADDING,
    circle=True,
) -> NDArray:
    if padding == 0:
        left = x - radius, y
        top = x, y - radius
        right = x + radius, y
        bottom = x, y + radius

        draw_line(image, left, right, color, thickness, line, shift)
        draw_line(image, top, bottom, color, thickness, line, shift)
    else:
        left1 = x - radius - padding, y
        left2 = x - padding, y

        top1 = x, y - radius - padding
        top2 = x, y - padding

        right1 = x + radius + padding, y
        right2 = x + padding, y

        bottom1 = x, y + radius + padding
        bottom2 = x, y + padding

        draw_line(image, left1, left2, color, thickness, line, shift)
        draw_line(image, top1, top2, color, thickness, line, shift)
        draw_line(image, right1, right2, color, thickness, line, shift)
        draw_line(image, bottom1, bottom2, color, thickness, line, shift)

    if circle:
        draw_circle_coord(image, x, y, radius, color, thickness, line, shift)

    return image


def draw_crosshair(
    image: NDArray,
    pos: PointN,
    radius=CROSSHAIR_POINT_RADIUS,
    thickness=CROSSHAIR_POINT_THICKNESS,
    color=CROSSHAIR_POINT_COLOR,
    line=CROSSHAIR_POINT_LINE_TYPE,
    shift=CROSSHAIR_POINT_SHIFT,
    padding=CROSSHAIR_POINT_PADDING,
    circle=True,
) -> NDArray:
    return draw_crosshair_coord(
        image, pos[0], pos[1], radius, thickness, color, line, shift, padding, circle
    )


class CvlDrawableCrosshair:
    @staticmethod
    def cvl_draw_crosshair_coord(
        image: NDArray,
        x: Number,
        y: Number,
        radius=CROSSHAIR_POINT_RADIUS,
        thickness=CROSSHAIR_POINT_THICKNESS,
        color=CROSSHAIR_POINT_COLOR,
        line=CROSSHAIR_POINT_LINE_TYPE,
        shift=CROSSHAIR_POINT_SHIFT,
        padding=CROSSHAIR_POINT_PADDING,
        circle=True,
    ):
        return draw_crosshair_coord(
            image, x, y, radius, thickness, color, line, shift, padding, circle
        )

    @staticmethod
    def cvl_draw_crosshair(
        image: NDArray,
        pos: PointN,
        radius=CROSSHAIR_POINT_RADIUS,
        thickness=CROSSHAIR_POINT_THICKNESS,
        color=CROSSHAIR_POINT_COLOR,
        line=CROSSHAIR_POINT_LINE_TYPE,
        shift=CROSSHAIR_POINT_SHIFT,
        padding=CROSSHAIR_POINT_PADDING,
        circle=True,
    ):
        return draw_crosshair(
            image, pos, radius, thickness, color, line, shift, padding, circle
        )

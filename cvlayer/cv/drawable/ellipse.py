# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

from cvlayer.cv.drawable.defaults import (
    DEFAULT_COLOR,
    DEFAULT_LINE_TYPE,
    DEFAULT_SHIFT,
    DEFAULT_THICKNESS,
)
from cvlayer.cv.types.color import normalize_color
from cvlayer.cv.types.line_type import normalize_line_type
from cvlayer.typing import Number, PointN, SizeN

DEFAULT_ELLIPSE_ANGLE_DEGREE: Final[float] = 0.0
DEFAULT_ELLIPSE_START_ANGLE_DEGREE: Final[float] = 0.0
DEFAULT_ELLIPSE_END_ANGLE_DEGREE: Final[float] = 360.0


def draw_ellipse_coord(
    image: NDArray,
    center_x: Number,
    center_y: Number,
    axes_x: Number,
    axes_y: Number,
    angle=DEFAULT_ELLIPSE_ANGLE_DEGREE,
    start_angle=DEFAULT_ELLIPSE_START_ANGLE_DEGREE,
    end_angle=DEFAULT_ELLIPSE_END_ANGLE_DEGREE,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    center = int(center_x), int(center_y)
    axes = int(axes_x), int(axes_y)
    _color = normalize_color(color)
    _line = normalize_line_type(line)
    return cv2.ellipse(
        image,
        center,
        axes,
        angle,
        start_angle,
        end_angle,
        _color,
        thickness,
        _line,
        shift,
    )


def draw_ellipse(
    image: NDArray,
    center: PointN,
    axes: SizeN,
    angle=DEFAULT_ELLIPSE_ANGLE_DEGREE,
    start_angle=DEFAULT_ELLIPSE_START_ANGLE_DEGREE,
    end_angle=DEFAULT_ELLIPSE_END_ANGLE_DEGREE,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    return draw_ellipse_coord(
        image,
        center[0],
        center[1],
        axes[0],
        axes[1],
        angle,
        start_angle,
        end_angle,
        color,
        thickness,
        line,
        shift,
    )


class CvlDrawableEllipse:
    @staticmethod
    def cvl_draw_ellipse_coord(
        image: NDArray,
        center_x: Number,
        center_y: Number,
        axes_x: Number,
        axes_y: Number,
        angle=DEFAULT_ELLIPSE_ANGLE_DEGREE,
        start_angle=DEFAULT_ELLIPSE_START_ANGLE_DEGREE,
        end_angle=DEFAULT_ELLIPSE_END_ANGLE_DEGREE,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        shift=DEFAULT_SHIFT,
    ):
        return draw_ellipse_coord(
            image,
            center_x,
            center_y,
            axes_x,
            axes_y,
            angle,
            start_angle,
            end_angle,
            color,
            thickness,
            line,
            shift,
        )

    @staticmethod
    def cvl_draw_ellipse(
        image: NDArray,
        center: PointN,
        axes: SizeN,
        angle=DEFAULT_ELLIPSE_ANGLE_DEGREE,
        start_angle=DEFAULT_ELLIPSE_START_ANGLE_DEGREE,
        end_angle=DEFAULT_ELLIPSE_END_ANGLE_DEGREE,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        shift=DEFAULT_SHIFT,
    ):
        return draw_ellipse(
            image,
            center,
            axes,
            angle,
            start_angle,
            end_angle,
            color,
            thickness,
            line,
            shift,
        )

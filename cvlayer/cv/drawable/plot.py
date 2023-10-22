# -*- coding: utf-8 -*-

from enum import Enum, auto, unique
from typing import Optional, Sequence

from numpy.typing import NDArray

from cvlayer.cv.drawable.defaults import (
    DEFAULT_COLOR,
    DEFAULT_LINE_TYPE,
    DEFAULT_RADIUS,
    DEFAULT_SHIFT,
    DEFAULT_THICKNESS,
)
from cvlayer.cv.drawable.line import draw_line
from cvlayer.cv.drawable.point import draw_point
from cvlayer.cv.drawable.rectangle import draw_rectangle
from cvlayer.typing import Number, PointN, RectI


@unique
class PlotMode(Enum):
    POINT = auto()
    LINE = auto()
    BAR_X = auto()
    BAR_Y = auto()


def draw_absolute_plot_points(
    canvas: NDArray,
    *points: PointN,
    radius=DEFAULT_RADIUS,
    color=DEFAULT_COLOR,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    for point in points:
        draw_point(canvas, point, radius, color, line, shift)
    return canvas


def draw_absolute_plot_lines(
    canvas: NDArray,
    *points: PointN,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    prev = points[0]
    for point in points[1:]:
        draw_line(canvas, prev, point, color, thickness, line, shift)
        prev = point
    return canvas


def draw_absolute_plot_x_bars(
    canvas: NDArray,
    *points: PointN,
    bottom: Number = 0,
    radius=DEFAULT_RADIUS,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    for point in points:
        x1 = point[0] - radius
        y1 = point[1]
        x2 = point[0] + radius
        y2 = bottom
        roi = x1, y1, x2, y2
        draw_rectangle(canvas, roi, color, thickness, line, shift)
    return canvas


def draw_absolute_plot_y_bars(
    canvas: NDArray,
    *points: PointN,
    left: Number = 0,
    radius=DEFAULT_RADIUS,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
) -> NDArray:
    for point in points:
        x1 = left
        y1 = point[1] - radius
        x2 = point[0]
        y2 = point[1] + radius
        roi = x1, y1, x2, y2
        draw_rectangle(canvas, roi, color, thickness, line, shift)
    return canvas


def draw_plot_2d(
    canvas: NDArray,
    *datasets: Sequence[Number],
    roi: Optional[RectI] = None,
    mode=PlotMode.POINT,
    color=DEFAULT_COLOR,
    thickness=DEFAULT_THICKNESS,
    line=DEFAULT_LINE_TYPE,
    shift=DEFAULT_SHIFT,
    radius=DEFAULT_RADIUS,
    min_x: Optional[Number] = None,
    max_x: Optional[Number] = None,
    min_y: Optional[Number] = None,
    max_y: Optional[Number] = None,
) -> NDArray:
    if len(datasets) != 2:
        raise ValueError("There must be 2 datasets")
    if datasets[0] == 0:
        raise ValueError("The datasets[0] must exist")
    if datasets[1] == 0:
        raise ValueError("The datasets[1] must exist")

    xs = datasets[0]
    ys = datasets[1]
    if len(xs) != len(ys):
        raise ValueError("The dataset size must be the same")

    min_x = min_x if min_x is not None else min(xs)
    max_x = max_x if max_x is not None else max(xs)
    min_y = min_y if min_y is not None else min(ys)
    max_y = max_y if max_y is not None else max(ys)
    assert min_x is not None
    assert max_x is not None
    assert min_y is not None
    assert max_y is not None
    x_size = max_x - min_x
    y_size = max_y - min_y
    if x_size * y_size == 0:
        minmax_roi = min_x, max_x, min_y, max_y
        raise ValueError(f"The data range is 0. roi is {minmax_roi}")

    roi = roi if roi else (0, 0, canvas.shape[1], canvas.shape[0])
    assert roi is not None

    left = min(roi[0], roi[2])
    right = max(roi[0], roi[2])
    top = min(roi[1], roi[3])
    bottom = max(roi[1], roi[3])
    width = right - left
    height = bottom - top
    if width * height == 0:
        raise ValueError("There is no area to draw")

    def canvas_x(real_x: Number):
        return left + (real_x / x_size * width)

    def canvas_y(real_y: Number):
        return bottom - (real_y / y_size * height)

    points = [(canvas_x(x), canvas_y(y)) for x, y in zip(xs, ys)]

    if mode == PlotMode.POINT:
        draw_absolute_plot_points(
            canvas,
            *points,
            radius=radius,
            color=color,
            line=line,
        )
    elif mode == PlotMode.LINE:
        draw_absolute_plot_lines(
            canvas,
            *points,
            color=color,
            thickness=thickness,
            line=line,
            shift=shift,
        )
    elif mode == PlotMode.BAR_X:
        draw_absolute_plot_x_bars(
            canvas,
            *points,
            bottom=bottom,
            radius=radius,
            color=color,
            thickness=thickness,
            line=line,
            shift=shift,
        )
    elif mode == PlotMode.BAR_Y:
        draw_absolute_plot_y_bars(
            canvas,
            *points,
            left=left,
            radius=radius,
            color=color,
            thickness=thickness,
            line=line,
            shift=shift,
        )
    else:
        raise ValueError(f"Unknown plot mode: {mode}")

    return canvas


class CvlDrawablePlot:
    @staticmethod
    def cvl_draw_plot_2d(
        canvas: NDArray,
        *datasets: Sequence[Number],
        roi: Optional[RectI] = None,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        shift=DEFAULT_SHIFT,
        mode=PlotMode.POINT,
        radius=DEFAULT_RADIUS,
        min_x: Optional[Number] = None,
        max_x: Optional[Number] = None,
        min_y: Optional[Number] = None,
        max_y: Optional[Number] = None,
    ) -> None:
        draw_plot_2d(
            canvas,
            *datasets,
            roi=roi,
            color=color,
            thickness=thickness,
            line=line,
            shift=shift,
            mode=mode,
            radius=radius,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )

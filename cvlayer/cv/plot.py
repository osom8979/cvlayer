# -*- coding: utf-8 -*-

from enum import Enum, auto, unique
from typing import Optional, Sequence

from numpy.typing import NDArray

from cvlayer.cv.drawable import (
    COLOR,
    LINE_TYPE,
    RADIUS,
    THICKNESS,
    draw_line,
    draw_point,
    draw_rectangle,
)
from cvlayer.typing import NumberT, PointT, RectInt


@unique
class PlotMode(Enum):
    POINT = auto()
    LINE = auto()
    BAR_X = auto()
    BAR_Y = auto()


def draw_absolute_plot_points(
    canvas: NDArray,
    radius=RADIUS,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
    *points: PointT,
) -> None:
    for point in points:
        draw_point(canvas, point[0], point[1], radius, color, thickness, line_type)


def draw_absolute_plot_lines(
    canvas: NDArray,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
    *points: PointT,
) -> None:
    prev = points[0]
    for point in points[1:]:
        draw_line(canvas, prev, point, color, thickness, line_type)
        prev = point


def draw_absolute_plot_x_bars(
    canvas: NDArray,
    bottom: NumberT,
    radius=RADIUS,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
    *points: PointT,
) -> None:
    for point in points:
        x1 = point[0] - radius
        y1 = point[1]
        x2 = point[0] + radius
        y2 = bottom
        roi = x1, y1, x2, y2
        draw_rectangle(canvas, roi, color, thickness, line_type)


def draw_absolute_plot_y_bars(
    canvas: NDArray,
    left: NumberT,
    radius=RADIUS,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
    *points: PointT,
) -> None:
    for point in points:
        x1 = left
        y1 = point[1] - radius
        x2 = point[0]
        y2 = point[1] + radius
        roi = x1, y1, x2, y2
        draw_rectangle(canvas, roi, color, thickness, line_type)


def draw_plot_2d(
    canvas: NDArray,
    *datasets: Sequence[NumberT],
    roi: Optional[RectInt] = None,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
    mode=PlotMode.POINT,
    radius=RADIUS,
    min_x: Optional[NumberT] = None,
    max_x: Optional[NumberT] = None,
    min_y: Optional[NumberT] = None,
    max_y: Optional[NumberT] = None,
) -> None:
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
        raise ValueError("The data range is 0")

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

    def canvas_x(real_x: NumberT):
        return left + (real_x / x_size * width)

    def canvas_y(real_y: NumberT):
        return bottom - (real_y / y_size * height)

    points = [(canvas_x(x), canvas_y(y)) for x, y in zip(xs, ys)]

    if mode == PlotMode.POINT:
        draw_absolute_plot_points(
            canvas,
            radius,
            color,
            thickness,
            line_type,
            *points,
        )
    elif mode == PlotMode.LINE:
        draw_absolute_plot_lines(
            canvas,
            color,
            thickness,
            line_type,
            *points,
        )
    elif mode == PlotMode.BAR_X:
        draw_absolute_plot_x_bars(
            canvas,
            bottom,
            radius,
            color,
            thickness,
            line_type,
            *points,
        )
    elif mode == PlotMode.BAR_Y:
        draw_absolute_plot_y_bars(
            canvas,
            left,
            radius,
            color,
            thickness,
            line_type,
            *points,
        )
    else:
        raise ValueError(f"Unknown plot mode: {mode}")


class CvlPlot:
    @staticmethod
    def cvl_draw_plot_2d(
        canvas: NDArray,
        *datasets: Sequence[NumberT],
        roi: Optional[RectInt] = None,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
        mode=PlotMode.POINT,
        radius=RADIUS,
        min_x: Optional[NumberT] = None,
        max_x: Optional[NumberT] = None,
        min_y: Optional[NumberT] = None,
        max_y: Optional[NumberT] = None,
    ) -> None:
        draw_plot_2d(
            canvas,
            *datasets,
            roi=roi,
            color=color,
            thickness=thickness,
            line_type=line_type,
            mode=mode,
            radius=radius,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )

# -*- coding: utf-8 -*-

from typing import Final, List, Optional, Sequence, Tuple

import cv2
from numpy import full, uint8
from numpy.typing import NDArray

from cvlayer.cv.color import PIXEL_8BIT_MAX
from cvlayer.cv.drawable import LINE_AA, draw_image, draw_line
from cvlayer.cv.plot import PlotMode, draw_plot_2d
from cvlayer.palette.basic import (
    AQUA,
    BLACK,
    BLUE,
    FUCHSIA,
    GRAY,
    GREEN,
    RED,
    WHITE,
    YELLOW,
)
from cvlayer.typing import Color, PointFloat, RectInt

RANGE_MAX: Final[int] = PIXEL_8BIT_MAX + 1
DEFAULT_HIST_SIZE: Final[Sequence[int]] = (RANGE_MAX,)
DEFAULT_RANGES: Final[Tuple[float, float]] = (0.0, float(RANGE_MAX))
THICKNESS: Final[int] = 1

BACKGROUND_COLOR: Final[Color] = WHITE
BACKGROUND_ALPHA: Final[float] = 0.4
PADDING: Final[int] = 12


def calc_hist(
    images: Sequence[NDArray],
    channels: Sequence[int],
    mask: Optional[NDArray] = None,
    hist_size: Sequence[int] = DEFAULT_HIST_SIZE,
    ranges: Sequence[float] = DEFAULT_RANGES,
    accumulate=False,
) -> NDArray:
    return cv2.calcHist(
        images=images,
        channels=channels,
        mask=mask,
        histSize=hist_size,
        ranges=ranges,
        accumulate=accumulate,
    )


def normalize_drawable_histogram(
    hist: NDArray,
    width: int,
    height: int,
    hist_size=RANGE_MAX,
) -> List[PointFloat]:
    normalized = hist.copy()
    cv2.normalize(
        src=hist,
        dst=normalized,
        alpha=0.0,
        beta=float(height),
        norm_type=cv2.NORM_MINMAX,
    )
    width_step = width / hist_size
    result = list()
    for i in range(hist_size):
        x = i * width_step
        y = float(normalized[i])
        result.append((x, y))
    return result


def draw_histogram_channel(
    canvas: NDArray,
    roi: RectInt,
    analysis: NDArray,
    analysis_roi: Optional[RectInt] = None,
    index=0,
    channel_max=float(RANGE_MAX),
    color=GRAY,
    thickness=THICKNESS,
    line_type=LINE_AA,
) -> None:
    width = abs(roi[2] - roi[0])
    height = abs(roi[3] - roi[1])
    hist_size = (height,)
    ranges = 0.0, channel_max
    if analysis_roi is not None:
        x1, y1, x2, y2 = analysis_roi
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        if (right - left) * (bottom - top) != 0:
            analysis = analysis[top:bottom, left:right]
    hist = calc_hist([analysis], [index], hist_size=hist_size, ranges=ranges)
    points = normalize_drawable_histogram(hist, width, height)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    draw_plot_2d(
        canvas,
        xs,
        ys,
        roi=roi,
        color=color,
        thickness=thickness,
        line_type=line_type,
        mode=PlotMode.LINE,
    )


def draw_histogram_channels(
    canvas: NDArray,
    roi: RectInt,
    analysis: NDArray,
    analysis_roi: Optional[RectInt] = None,
    channels_max: Sequence[float] = (RANGE_MAX, RANGE_MAX, RANGE_MAX),
    colors: Optional[Sequence[Color]] = None,
    thickness=THICKNESS,
    line_type=LINE_AA,
) -> None:
    if analysis_roi is not None:
        x1, y1, x2, y2 = analysis_roi
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        if (right - left) * (bottom - top) != 0:
            analysis = analysis[top:bottom, left:right]

    shape_size = len(analysis.shape)
    if shape_size == 2:
        channels = 1
    elif shape_size == 3:
        channels = analysis.shape[2]
    else:
        raise ValueError(f"Unsupported analysis shape: {shape_size}")

    if colors:
        channels_color = colors
    else:
        if channels == 1:
            channels_color = (BLACK,)
        elif channels == 2:
            channels_color = AQUA, YELLOW
        elif channels == 3:
            channels_color = BLUE, GREEN, RED
        elif channels == 4:
            channels_color = BLUE, GREEN, RED, FUCHSIA
        else:
            raise ValueError(f"Unsupported channels: {channels}")

    for i in range(channels):
        draw_histogram_channel(
            canvas,
            roi,
            analysis,
            None,
            i,
            channels_max[i],
            channels_color[i],
            thickness,
            line_type,
        )


def draw_histogram_channels_with_decorate(
    canvas: NDArray,
    roi: RectInt,
    analysis: NDArray,
    analysis_roi: Optional[RectInt] = None,
    channels_max: Sequence[float] = (RANGE_MAX, RANGE_MAX, RANGE_MAX),
    colors: Optional[Sequence[Color]] = None,
    thickness=THICKNESS,
    line_type=LINE_AA,
    background_color=BACKGROUND_COLOR,
    background_alpha=BACKGROUND_ALPHA,
    padding=PADDING,
) -> None:
    x1, y1, x2, y2 = roi
    box_left = min(x1, x2)
    box_right = max(x1, x2)
    box_top = min(y1, y2)
    box_bottom = max(y1, y2)
    box_width = box_right - box_left
    box_height = box_bottom - box_top
    assert box_width >= 1
    assert box_height >= 1
    box = full((box_height, box_width, 3), background_color, dtype=uint8)

    plot_left = padding
    plot_top = padding
    plot_right = box_width - padding
    plot_bottom = box_height - padding
    plot_canvas_roi = plot_left, plot_top, plot_right, plot_bottom

    left_top = plot_left, plot_top
    left_bottom = plot_left, plot_bottom
    right_bottom = plot_right, plot_bottom
    draw_line(box, left_bottom, right_bottom, BLACK, 1)
    draw_line(box, left_bottom, left_top, BLACK, 1)

    center_x_bottom0 = plot_left + (box_width // 2), plot_bottom
    center_x_bottom1 = center_x_bottom0[0], plot_bottom + (padding // 2)
    draw_line(box, center_x_bottom0, center_x_bottom1, BLACK, thickness=thickness)

    left_center_y0 = plot_left, plot_top + (box_height // 2)
    left_center_y1 = plot_left - (padding // 2), left_center_y0[1]
    draw_line(box, left_center_y0, left_center_y1, BLACK, thickness=thickness)

    draw_histogram_channels(
        canvas=box,
        roi=plot_canvas_roi,
        analysis=analysis,
        analysis_roi=analysis_roi,
        channels_max=channels_max,
        colors=colors,
        thickness=thickness,
        line_type=line_type,
    )

    draw_image(canvas, box, box_left, box_top, background_alpha)


class CvlHistogram:
    @staticmethod
    def cvl_calc_hist(
        images: Sequence[NDArray],
        channels: Sequence[int],
        mask: Optional[NDArray] = None,
        hist_size: Sequence[int] = DEFAULT_HIST_SIZE,
        ranges: Sequence[float] = DEFAULT_RANGES,
        accumulate=False,
    ):
        return calc_hist(
            images=images,
            channels=channels,
            mask=mask,
            hist_size=hist_size,
            ranges=ranges,
            accumulate=accumulate,
        )

    @staticmethod
    def cvl_normalize_drawable_histogram(
        hist: NDArray,
        width: int,
        height: int,
        hist_size=RANGE_MAX,
    ):
        return normalize_drawable_histogram(
            hist=hist,
            width=width,
            height=height,
            hist_size=hist_size,
        )

    @staticmethod
    def cvl_draw_histogram_channel(
        canvas: NDArray,
        roi: RectInt,
        analysis: NDArray,
        analysis_roi: Optional[RectInt] = None,
        index=0,
        channel_max=RANGE_MAX,
        color=GRAY,
        thickness=THICKNESS,
        line_type=LINE_AA,
    ):
        return draw_histogram_channel(
            canvas=canvas,
            roi=roi,
            analysis=analysis,
            analysis_roi=analysis_roi,
            index=index,
            channel_max=channel_max,
            color=color,
            thickness=thickness,
            line_type=line_type,
        )

    @staticmethod
    def cvl_draw_histogram_channels(
        canvas: NDArray,
        roi: RectInt,
        analysis: NDArray,
        analysis_roi: Optional[RectInt] = None,
        channels_max=(RANGE_MAX, RANGE_MAX, RANGE_MAX),
        colors: Optional[Sequence[Color]] = None,
        thickness=THICKNESS,
        line_type=LINE_AA,
    ):
        return draw_histogram_channels(
            canvas=canvas,
            roi=roi,
            analysis=analysis,
            analysis_roi=analysis_roi,
            channels_max=channels_max,
            colors=colors,
            thickness=thickness,
            line_type=line_type,
        )

    @staticmethod
    def cvl_draw_histogram_channels_with_decorate(
        canvas: NDArray,
        roi: RectInt,
        analysis: NDArray,
        analysis_roi: Optional[RectInt] = None,
        channels_max: Sequence[float] = (RANGE_MAX, RANGE_MAX, RANGE_MAX),
        colors: Optional[Sequence[Color]] = None,
        thickness=THICKNESS,
        line_type=LINE_AA,
        background_color=BACKGROUND_COLOR,
        background_alpha=BACKGROUND_ALPHA,
        padding=PADDING,
    ):
        return draw_histogram_channels_with_decorate(
            canvas=canvas,
            roi=roi,
            analysis=analysis,
            analysis_roi=analysis_roi,
            channels_max=channels_max,
            colors=colors,
            thickness=thickness,
            line_type=line_type,
            background_color=background_color,
            background_alpha=background_alpha,
            padding=padding,
        )

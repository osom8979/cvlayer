# -*- coding: utf-8 -*-

from typing import Final, List, Optional, Sequence, Tuple

import cv2
from numpy.typing import NDArray

from cvlayer.cv.color import PIXEL_8BIT_MAX
from cvlayer.cv.drawable import LINE_AA
from cvlayer.cv.plot import PlotMode, draw_plot_2d
from cvlayer.palette.basic import BLUE, GRAY, GREEN, RED
from cvlayer.typing import Color, PointFloat, RectInt

RANGE_MAX: Final[int] = PIXEL_8BIT_MAX + 1
DEFAULT_HIST_SIZE: Final[Sequence[int]] = (RANGE_MAX,)
DEFAULT_RANGES: Final[Tuple[float, float]] = (0.0, float(RANGE_MAX))


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
        y = height - float(normalized[i])
        result.append((x, y))
    return result


def draw_histogram_channel(
    frame: NDArray,
    roi: RectInt,
    analysis: NDArray,
    analysis_roi: Optional[RectInt] = None,
    index=0,
    channel_max=float(RANGE_MAX),
    color=GRAY,
    thickness=1,
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
        frame,
        xs,
        ys,
        roi=roi,
        color=color,
        thickness=thickness,
        line_type=line_type,
        mode=PlotMode.LINE,
    )


def draw_histogram_channels(
    frame: NDArray,
    roi: RectInt,
    analysis: NDArray,
    analysis_roi: Optional[RectInt] = None,
    channels_max: Sequence[float] = (RANGE_MAX, RANGE_MAX, RANGE_MAX),
    colors: Sequence[Color] = (BLUE, GREEN, RED),
    thickness=1,
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

    for i in range(channels):
        draw_histogram_channel(
            frame,
            roi,
            analysis,
            None,
            i,
            channels_max[i],
            colors[i],
            thickness,
            line_type,
        )


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
        frame: NDArray,
        roi: RectInt,
        analysis: NDArray,
        analysis_roi: Optional[RectInt] = None,
        index=0,
        channel_max=RANGE_MAX,
        color=GRAY,
        thickness=1,
        line_type=LINE_AA,
    ):
        return draw_histogram_channel(
            frame=frame,
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
        frame: NDArray,
        roi: RectInt,
        analysis: NDArray,
        analysis_roi: Optional[RectInt] = None,
        channels_max=(RANGE_MAX, RANGE_MAX, RANGE_MAX),
        colors=(BLUE, GREEN, RED),
        thickness=1,
        line_type=LINE_AA,
    ):
        return draw_histogram_channels(
            frame=frame,
            roi=roi,
            analysis=analysis,
            analysis_roi=analysis_roi,
            channels_max=channels_max,
            colors=colors,
            thickness=thickness,
            line_type=line_type,
        )

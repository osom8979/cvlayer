# -*- coding: utf-8 -*-

from typing import Final, List, Optional, Sequence, Tuple

import cv2
from numpy import float32, uint8, zeros
from numpy.typing import NDArray

from cvlayer.cv.color import PIXEL_8BIT_MAX
from cvlayer.palette.basic import BLUE, GREEN, RED
from cvlayer.typing import Color, RectInt

RANGE_MAX: Final[int] = PIXEL_8BIT_MAX + 1
DEFAULT_BINS: Final[Sequence[int]] = (RANGE_MAX,)
DEFAULT_RANGES: Final[Tuple[int, int]] = (0, RANGE_MAX)

DEFAULT_HISTOGRAM_COLORS = (BLUE, GREEN, RED)
DEFAULT_THICKNESS: Final[int] = 1
DEFAULT_LINE_TYPE: Final[int] = cv2.LINE_AA


class Histogram:
    _mask: Optional[NDArray]
    _histograms: List[NDArray]

    def __init__(
        self,
        bins=DEFAULT_BINS,
        ranges=DEFAULT_RANGES,
    ):
        self._bins = bins
        self._ranges = ranges
        self._mask = None
        self._histograms = list()

    @property
    def histograms(self) -> List[NDArray]:
        return self._histograms

    def calc_hist(self, frame: NDArray, channel: int) -> NDArray:
        return cv2.calcHist([frame], [channel], self._mask, self._bins, self._ranges)

    def draw(
        self,
        frame: NDArray,
        colors: Sequence[Color] = DEFAULT_HISTOGRAM_COLORS,
        thickness=DEFAULT_THICKNESS,
        line_type=DEFAULT_LINE_TYPE,
    ) -> None:
        height, width = frame.shape[0:2]
        assert len(self._histograms) in (1, 3)

        hist_size = self._bins[0]
        bin_w = int(round(width / hist_size))

        for histogram, color in zip(self._histograms, colors):
            hist = cv2.normalize(
                histogram,
                None,
                alpha=0,
                beta=height,
                norm_type=cv2.NORM_MINMAX,
            )

            for i in range(1, hist_size):
                x1 = bin_w * (i - 1)
                y1 = height - int(hist[i - 1])
                x2 = bin_w * i
                y2 = height - int(hist[i])
                p1 = x1, y1
                p2 = x2, y2
                cv2.line(frame, p1, p2, color, thickness, line_type)

    def calc(self, frame: NDArray, roi: Optional[RectInt] = None) -> None:
        assert frame.dtype in (uint8, float32)
        assert len(frame.shape) in (2, 3)

        if roi is not None:
            x1, y1, x2, y2 = roi
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            area = abs(right - left) * abs(bottom - top)

            if area > 0:
                self._mask = zeros(frame.shape[:2], uint8)
                self._mask[top:bottom, left:right] = 255
            else:
                self._mask = None
        else:
            self._mask = None

        channels = 1 if len(frame.shape) == 2 else frame.shape[2]
        self._histograms = [self.calc_hist(frame, i) for i in range(channels)]


class CvlHistogram:
    @staticmethod
    def cvl_create_histogram(bins=DEFAULT_BINS, ranges=DEFAULT_RANGES):
        return Histogram(bins, ranges)

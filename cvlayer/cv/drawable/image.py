# -*- coding: utf-8 -*-

from typing import Final

import cv2
from numpy.typing import NDArray

from cvlayer.typing import Number, PointN

ALPHA_MIN: Final[float] = 0
ALPHA_MAX: Final[float] = 1.0

DEFAULT_DRAW_IMAGE_X: Final[Number] = 0
DEFAULT_DRAW_IMAGE_Y: Final[Number] = 0
DEFAULT_DRAW_IMAGE_POS: Final[PointN] = DEFAULT_DRAW_IMAGE_X, DEFAULT_DRAW_IMAGE_Y
DEFAULT_DRAW_IMAGE_ALPHA: Final[float] = ALPHA_MAX


def draw_image_coord(
    canvas: NDArray,
    src: NDArray,
    x=DEFAULT_DRAW_IMAGE_X,
    y=DEFAULT_DRAW_IMAGE_Y,
    alpha=DEFAULT_DRAW_IMAGE_ALPHA,
) -> NDArray:
    if alpha == ALPHA_MIN:
        return canvas

    if alpha < ALPHA_MIN:
        raise ValueError("The 'alpha' argument must be greater than or equal to 0")
    if alpha > ALPHA_MAX:
        raise ValueError("The 'alpha' argument must not exceed 1")

    canvas_height = canvas.shape[0]
    canvas_width = canvas.shape[1]
    src_height = src.shape[0]
    src_width = src.shape[1]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(x1 + src_width, canvas_width)
    y2 = min(y1 + src_height, canvas_height)
    w = x2 - x1
    h = y2 - y1
    if w == 0 or h == 0:
        return canvas

    assert w > 0
    assert h > 0

    if w != src_width or h != src_height:
        src = src[0:h, 0:w]

    if alpha == ALPHA_MAX:
        canvas[y1:y2, x1:x2] = src
        return canvas

    assert ALPHA_MIN < alpha < ALPHA_MAX
    beta = ALPHA_MAX - alpha
    mixed = cv2.addWeighted(src, alpha, canvas[y1:y2, x1:x2], beta, 0)
    canvas[y1:y2, x1:x2] = mixed
    return canvas


def draw_image(
    canvas: NDArray,
    src: NDArray,
    pos=DEFAULT_DRAW_IMAGE_POS,
    alpha=DEFAULT_DRAW_IMAGE_ALPHA,
) -> NDArray:
    return draw_image_coord(canvas, src, pos[0], pos[1], alpha)


class CvlDrawableImage:
    @staticmethod
    def cvl_draw_image_coord(
        canvas: NDArray,
        src: NDArray,
        x=DEFAULT_DRAW_IMAGE_X,
        y=DEFAULT_DRAW_IMAGE_Y,
        alpha=DEFAULT_DRAW_IMAGE_ALPHA,
    ):
        return draw_image_coord(canvas, src, x, y, alpha)

    @staticmethod
    def cvl_draw_image(
        canvas: NDArray,
        src: NDArray,
        pos=DEFAULT_DRAW_IMAGE_POS,
        alpha=DEFAULT_DRAW_IMAGE_ALPHA,
    ):
        return draw_image(canvas, src, pos, alpha)

# -*- coding: utf-8 -*-

from typing import Final

from numpy import nonzero, uint8, zeros
from numpy.typing import NDArray

from cvlayer.cv.border import copy_make_border
from cvlayer.cv.types.border import BorderType

DEFAULT_KERNEL_X: Final[int] = 7
DEFAULT_KERNEL_Y: Final[int] = 7


try:
    from numba import njit, prange  # noqa

    @njit(parallel=True)
    def _color_quantization(
        mask: NDArray[uint8],
        gray: NDArray[uint8],
        canvas: NDArray[uint8],
        kx: int,
        ky: int,
    ) -> NDArray[uint8]:
        kx2 = kx * 2
        ky2 = ky * 2
        yis, xis = nonzero(mask)
        assert yis.size == xis.size
        for i in prange(yis.size):
            cx = xis[i]
            cy = yis[i]
            x1 = cx  # == `cx - kx + kx`
            y1 = cy  # == `cy - ky + ky`
            x2 = cx + kx2  # == `cx + kx + kx`
            y2 = cy + ky2  # == `cy + ky + ky`
            canvas[cy, cx] = gray[y1:y2, x1:x2].mean()
        return canvas

except ImportError:

    def _color_quantization(
        mask: NDArray[uint8],
        gray: NDArray[uint8],
        canvas: NDArray[uint8],
        kx: int,
        ky: int,
    ) -> NDArray[uint8]:
        kx2 = kx * 2
        ky2 = ky * 2
        yis, xis = nonzero(mask)
        assert yis.size == xis.size
        for i in range(yis.size):
            cx = xis[i]
            cy = yis[i]
            x1 = cx  # == `cx - kx + kx`
            y1 = cy  # == `cy - ky + ky`
            x2 = cx + kx2  # == `cx + kx + kx`
            y2 = cy + ky2  # == `cy + ky + ky`
            canvas[cy, cx] = gray[y1:y2, x1:x2].mean()
        return canvas


def mean_shift_color_quantization(
    mask: NDArray[uint8],
    gray: NDArray[uint8],
    kx=DEFAULT_KERNEL_X,
    ky=DEFAULT_KERNEL_Y,
) -> NDArray[uint8]:
    border_gray = copy_make_border(gray, ky, ky, kx, kx, BorderType.REFLECT101)
    height, width = gray.shape[0:2]
    canvas = zeros((height, width), dtype=uint8)
    return _color_quantization(mask, border_gray, canvas, kx, ky)

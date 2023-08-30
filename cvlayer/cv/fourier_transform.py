# -*- coding: utf-8 -*-

from typing import Final

from numpy import abs, uint8
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from numpy.typing import NDArray

DEFAULT_LOW_FREQUENCY_BOUNDARY: Final[int] = 10


def fast_fourier_transform(src: NDArray, low=DEFAULT_LOW_FREQUENCY_BOUNDARY):
    f_shift = fftshift(fft2(src))

    rows, cols = src.shape[0:2]
    row_half, col_half = rows // 2, cols // 2

    r_begin = row_half - low
    r_end = row_half + low
    c_begin = col_half - low
    c_end = col_half + low

    f_shift[r_begin:r_end, c_begin:c_end] = 1

    return uint8(abs(ifft2(ifftshift(f_shift))))


class CvlFourierTransform:
    @staticmethod
    def cvl_fast_fourier_transform(src: NDArray, low=DEFAULT_LOW_FREQUENCY_BOUNDARY):
        return fast_fourier_transform(src, low)

# -*- coding: utf-8 -*-

from numpy import uint8
from numpy.typing import NDArray


def clip_vertical_section(
    mask: NDArray[uint8],
    top: int,
    bottom: int,
) -> NDArray[uint8]:
    result = mask.copy()
    result[0:top, :] = 0
    result[bottom:, :] = 0
    return result


def clip_horizontal_section(
    mask: NDArray[uint8],
    left: int,
    right: int,
) -> NDArray[uint8]:
    result = mask.copy()
    result[:, 0:left] = 0
    result[:, right:] = 0
    return result


def clip_section(
    mask: NDArray[uint8],
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> NDArray[uint8]:
    result = mask.copy()
    result[:, 0:left] = 0
    result[:, right:] = 0
    result[0:top, :] = 0
    result[bottom:, :] = 0
    return result

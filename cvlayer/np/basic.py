# -*- coding: utf-8 -*-

from numpy.typing import NDArray


def image_mins(array: NDArray):
    shape = array.shape
    if len(shape) == 2:
        return [array.min()]
    elif len(shape) == 3:
        print(array)
        return [array[:, :, c].min() for c in range(shape[2])]
    else:
        raise ValueError(f"Unsupported array shape: {shape}")


def image_maxs(array: NDArray):
    shape = array.shape
    if len(shape) == 2:
        return [array.max()]
    elif len(shape) == 3:
        return [array[:, :, c].max() for c in range(shape[2])]
    else:
        raise ValueError(f"Unsupported array shape: {shape}")


def image_means(array: NDArray):
    shape = array.shape
    if len(shape) == 2:
        return [array.mean()]
    elif len(shape) == 3:
        return [array[:, :, c].mean() for c in range(shape[2])]
    else:
        raise ValueError(f"Unsupported array shape: {shape}")

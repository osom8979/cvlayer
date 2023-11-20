# -*- coding: utf-8 -*-

from typing import Sequence, Union

from cvlayer.palette import css4_palette

Color = Sequence[float]
ColorLike = Union[int, float, str, Sequence[int], Sequence[float]]


def normalize_color(color: ColorLike) -> Color:
    if isinstance(color, (int, float)):
        return (color,)
    elif isinstance(color, str):
        return css4_palette()[color.upper()]
    elif isinstance(color, Sequence):
        # Do not use `tuple(c for c in color)`. It's too slow.
        return color  # type: ignore[return-value]
    else:
        raise TypeError(f"Unsupported color type: {type(color).__name__}")

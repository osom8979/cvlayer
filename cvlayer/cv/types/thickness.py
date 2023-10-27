# -*- coding: utf-8 -*-

from typing import Final, Literal, Union

import cv2

FILLED: Final[int] = cv2.FILLED


class Filled:
    pass


ThicknessLike = Union[Filled, int, Literal[-1]]

assert FILLED == -1


def normalize_thickness(thickness: ThicknessLike) -> int:
    if isinstance(thickness, Filled):
        return FILLED
    elif isinstance(thickness, int):
        return thickness
    else:
        raise TypeError(f"Unsupported thickness type: {type(thickness).__name__}")

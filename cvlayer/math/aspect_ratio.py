# -*- coding: utf-8 -*-

from math import gcd
from typing import Optional, Tuple

from cvlayer.typing import NumberT


def aspect_ratio(a: int, b: int) -> Tuple[int, int]:
    g = gcd(a, b)
    return a // g, b // g


def rescale_aspect_ratio(
    x: NumberT,
    y: NumberT,
    dx: Optional[NumberT] = None,
    dy: Optional[NumberT] = None,
) -> Tuple[NumberT, NumberT]:
    assert type(x) is type(y)

    if dx is not None and dy is not None:
        return dx, dy

    if dx is None and dy is None:
        return x, y

    elif dx is None:
        assert dy is not None
        return type(x)(x * dy / y), dy
    else:
        assert dy is None
        return dx, type(y)(y * dx / x)

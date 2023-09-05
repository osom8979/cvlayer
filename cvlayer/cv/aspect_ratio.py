# -*- coding: utf-8 -*-

from typing import Optional

from cvlayer.math.aspect_ratio import aspect_ratio, rescale_aspect_ratio
from cvlayer.typing import NumberT


class CvlAspectRatio:
    @staticmethod
    def cvl_aspect_ratio(a: int, b: int):
        return aspect_ratio(a, b)

    @staticmethod
    def rescale_aspect_ratio(
        x: NumberT,
        y: NumberT,
        dx: Optional[NumberT] = None,
        dy: Optional[NumberT] = None,
    ):
        return rescale_aspect_ratio(x, y, dx, dy)

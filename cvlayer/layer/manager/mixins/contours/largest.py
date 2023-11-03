# -*- coding: utf-8 -*-

from typing import Optional, Sequence

from numpy import int32
from numpy.typing import NDArray

from cvlayer.cv.contour.find import find_largest_contour_index
from cvlayer.cv.drawable.contours import draw_contour
from cvlayer.cv.drawable.defaults import (
    DEFAULT_COLOR,
    DEFAULT_LINE_TYPE,
    DEFAULT_THICKNESS,
)
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase
from cvlayer.typing import PointI


class CvmContoursLargest(LayerManagerMixinBase):
    def cvm_contours_largest(
        self,
        name: str,
        oriented=False,
        canvas: Optional[NDArray] = None,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        offset: Optional[PointI] = None,
        frame: Optional[NDArray] = None,
        contours: Optional[Sequence[NDArray[int32]]] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            data = contours if contours is not None else layer.prev_data

            index = find_largest_contour_index(data, oriented)
            result = data[index]

            if canvas is None:
                canvas = src.copy()

            layer.frame = draw_contour(
                canvas,
                result,
                color,
                thickness,
                line,
                offset,
            )
            layer.data = result

        return canvas, result

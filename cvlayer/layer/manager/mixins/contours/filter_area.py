# -*- coding: utf-8 -*-

from functools import partial
from typing import NamedTuple, Optional, Sequence

from numpy import int32
from numpy.typing import NDArray

from cvlayer.cv.contour.analysis import contour_area
from cvlayer.cv.drawable.contours import DRAW_ALL_CONTOURS, draw_contours
from cvlayer.cv.drawable.defaults import (
    DEFAULT_COLOR,
    DEFAULT_LINE_TYPE,
    DEFAULT_THICKNESS,
)
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase
from cvlayer.math.climit import INT_MAX
from cvlayer.typing import PointI


class _ContourArea(NamedTuple):
    contour: NDArray
    area: float


def _make_contour_area(contour: NDArray, oriented: bool) -> _ContourArea:
    return _ContourArea(contour, contour_area(contour, oriented))


def _area_filter(contour: _ContourArea, area_min: float, area_max: float) -> bool:
    if area_min != 0.0 and contour.area < area_min:
        return False
    if area_max != 0.0 and area_max < contour.area:
        return False
    return True


class CvmContoursFilterArea(LayerManagerMixinBase):
    def cvm_contours_filter_area(
        self,
        name: str,
        area_min=0.0,
        area_max=0.0,
        step=10.0,
        oriented=False,
        canvas: Optional[NDArray] = None,
        index=DRAW_ALL_CONTOURS,
        color=DEFAULT_COLOR,
        thickness=DEFAULT_THICKNESS,
        line=DEFAULT_LINE_TYPE,
        max_level=INT_MAX,
        offset: Optional[PointI] = None,
        frame: Optional[NDArray] = None,
        contours: Optional[Sequence[NDArray[int32]]] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            data = contours if contours is not None else layer.prev_data

            if not isinstance(data, Sequence):
                raise TypeError(f"Not a sequence data type: {type(data).__name__}")

            amin = layer.param("amin").build_float(area_min, 0.0, step=step).value
            amax = layer.param("amax").build_float(area_max, 0.0, step=step).value
            idx = layer.param("index").build_int(index, DRAW_ALL_CONTOURS).value

            cas1 = map(lambda x: _make_contour_area(x, oriented), data)
            cas2 = filter(partial(_area_filter, area_min=amin, area_max=amax), cas1)
            result = [ca.contour for ca in cas2]

            if canvas is None:
                canvas = src.copy()

            layer.frame = draw_contours(
                canvas,
                data,
                idx,
                color,
                thickness,
                line,
                None,
                max_level,
                offset,
            )
            layer.data = result

        return canvas, result

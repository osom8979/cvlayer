# -*- coding: utf-8 -*-

from enum import Enum, auto, unique
from typing import Dict, NamedTuple, Optional

from numpy import ndarray, uint8, zeros
from numpy.typing import NDArray

from cvlayer.cv.contour.find import contour_area, find_contours
from cvlayer.cv.contour.moments import moments
from cvlayer.cv.cvt_color import cvt_color_GRAY2BGR
from cvlayer.cv.drawable.contours import draw_contour
from cvlayer.cv.drawable.point import draw_point
from cvlayer.cv.drawable.text.outline import draw_outline_text
from cvlayer.cv.types.chain_approx import ChainApproximation
from cvlayer.cv.types.retrieval import Retrieval
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase
from cvlayer.palette import xkcd_palette
from cvlayer.typing import Color

_CCOMP = Retrieval.CCOMP
_EXTERNAL = Retrieval.EXTERNAL
_LIST = Retrieval.LIST
_TREE = Retrieval.TREE
_FLOODFILL = Retrieval.FLOODFILL

_NONE = ChainApproximation.NONE
_SIMPLE = ChainApproximation.SIMPLE
_TC89_KCOS = ChainApproximation.TC89_KCOS
_TC89_L1 = ChainApproximation.TC89_L1


@unique
class ContourSelectMode(Enum):
    ALL = auto()
    INDEX = auto()
    LARGEST = auto()
    SMALLEST = auto()
    LEFT_MOST = auto()
    RIGHT_MOST = auto()
    TOP_MOST = auto()
    BOTTOM_MOST = auto()


class _ContourArea(NamedTuple):
    contour: NDArray
    area: float


def _make_contour_area(contour: NDArray, oriented=False) -> _ContourArea:
    return _ContourArea(contour, contour_area(contour, oriented=oriented))


def _area_filter(contour: _ContourArea, amin: float, amax: float) -> bool:
    if amin != 0.0 and contour.area < amin:
        return False
    if amax != 0.0 and amax < contour.area:
        return False
    return True


class CvmContoursFind(LayerManagerMixinBase):
    def _cvm_find_contours(
        self,
        name: str,
        mode: Retrieval,
        method: ChainApproximation,
        area_min: float,
        area_max: float,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            if canvas is None:
                height, width = src.shape
                canvas = zeros((height, width, 3), dtype=uint8)
            else:
                assert canvas is not None
                if len(canvas.shape) == 2:
                    canvas = cvt_color_GRAY2BGR(canvas)

            if palette is None:
                palette = xkcd_palette()

            assert isinstance(canvas, ndarray)
            assert isinstance(palette, dict)

            md = layer.param("mode").build_enum(mode).value
            mt = layer.param("method").build_enum(method).value
            amin = layer.param("amin").build_float(area_min, 0.0, step=100.0).value
            amax = layer.param("amax").build_float(area_max, 0.0, step=100.0).value
            contours = list(find_contours(src, md, mt)[0])
            filtered_contours = list()
            for contour, color in zip(contours, palette.values()):
                area = contour_area(contour)
                if area < amin or amax < area:
                    continue

                m = moments(contour)
                if m.m00 == 0:
                    continue

                center = m.center
                draw_contour(canvas, contour, color)
                draw_point(canvas, center, color=color)
                draw_outline_text(canvas, f"{area:.2f}", center, color=color)
                filtered_contours.append(contour)

            layer.frame = canvas
            layer.data = filtered_contours
        return canvas, filtered_contours

    def cvm_find_contours_ccomp_none(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _CCOMP, _NONE, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_ccomp_simple(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _CCOMP, _SIMPLE, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_ccomp_tc89_kcos(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _CCOMP, _TC89_KCOS, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_ccomp_tc89_l1(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _CCOMP, _TC89_L1, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_external_none(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _EXTERNAL, _NONE, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_external_simple(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _EXTERNAL, _SIMPLE, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_external_tc89_kcos(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _EXTERNAL, _TC89_KCOS, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_external_tc89_l1(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _EXTERNAL, _TC89_L1, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_list_none(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _LIST, _NONE, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_list_simple(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _LIST, _SIMPLE, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_list_tc89_kcos(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _LIST, _TC89_KCOS, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_list_tc89_l1(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _LIST, _TC89_L1, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_tree_none(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _TREE, _NONE, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_tree_simple(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _TREE, _SIMPLE, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_tree_tc89_kcos(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _TREE, _TC89_KCOS, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_tree_tc89_l1(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _TREE, _TC89_L1, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_floodfill_none(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _FLOODFILL, _NONE, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_floodfill_simple(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _FLOODFILL, _SIMPLE, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_floodfill_tc89_kcos(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _FLOODFILL, _TC89_KCOS, area_min, area_max, canvas, palette, frame
        )

    def cvm_find_contours_floodfill_tc89_l1(
        self,
        name: str,
        area_min=1_000.0,
        area_max=500_000.0,
        canvas: Optional[NDArray] = None,
        palette: Optional[Dict[str, Color]] = None,
        frame: Optional[NDArray] = None,
    ):
        return self._cvm_find_contours(
            name, _FLOODFILL, _TC89_L1, area_min, area_max, canvas, palette, frame
        )

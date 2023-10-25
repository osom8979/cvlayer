# -*- coding: utf-8 -*-

from functools import partial
from typing import Final, Optional

from numpy.typing import NDArray

from cvlayer.cv.morphology import (
    MorphOperator,
    MorphShape,
    dilate,
    erode,
    get_structuring_element,
    morphology_ex,
)
from cvlayer.layer.base import LayerBase
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase

_SHAPE_PARAMETER_KEY: Final[str] = "shape"

_RECT = MorphShape.RECT
_ELLIPSE = MorphShape.ELLIPSE
_CROSS = MorphShape.CROSS

_ERODE = MorphOperator.ERODE
_DILATE = MorphOperator.DILATE
_OPEN = MorphOperator.OPEN
_CLOSE = MorphOperator.CLOSE
_GRADIENT = MorphOperator.GRADIENT
_TOPHAT = MorphOperator.TOPHAT
_BLACKHAT = MorphOperator.BLACKHAT
_HITMISS = MorphOperator.HITMISS


def _gse(layer: LayerBase, _old, _new):
    shape_param = layer.param(_SHAPE_PARAMETER_KEY)
    assert shape_param.initialized
    shape = shape_param.value
    assert isinstance(shape, MorphShape)
    assert isinstance(_new, int)
    return get_structuring_element(shape, (_new, _new))


class CvmMorphologyErode(LayerManagerMixinBase):
    def _cvm_erode(
        self,
        name: str,
        k: int,
        i: int,
        shape: MorphShape,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            layer.param(_SHAPE_PARAMETER_KEY).build_enum(shape)
            m = layer.param("m").build_uint(k, 1, cacher=partial(_gse, layer)).cache
            i = layer.param("i").build_uint(i, 1).value
            ax = layer.param("anchor_x").build_int(-1).value
            ay = layer.param("anchor_y").build_int(-1).value
            src = frame if frame is not None else layer.prev_frame
            result = erode(src, m, (ax, ay), i)
            layer.frame = result
        return result

    def cvm_erode_rect(self, name: str, k=3, i=1, frame: Optional[NDArray] = None):
        return self._cvm_erode(name, k, i, _RECT, frame)

    def cvm_erode_cross(self, name: str, k=3, i=1, frame: Optional[NDArray] = None):
        return self._cvm_erode(name, k, i, _CROSS, frame)

    def cvm_erode_ellipse(self, name: str, k=3, i=1, frame: Optional[NDArray] = None):
        return self._cvm_erode(name, k, i, _ELLIPSE, frame)


class CvmMorphologyDilate(LayerManagerMixinBase):
    def _cvm_dilate(
        self,
        name: str,
        k: int,
        i: int,
        shape: MorphShape,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            layer.param(_SHAPE_PARAMETER_KEY).build_enum(shape)
            m = layer.param("m").build_uint(k, 1, cacher=partial(_gse, layer)).cache
            i = layer.param("i").build_uint(i, 1).value
            ax = layer.param("anchor_x").build_int(-1).value
            ay = layer.param("anchor_y").build_int(-1).value
            src = frame if frame is not None else layer.prev_frame
            result = dilate(src, m, (ax, ay), i)
            layer.frame = result
        return result

    def cvm_dilate_rect(self, name: str, k=3, i=1, frame: Optional[NDArray] = None):
        return self._cvm_dilate(name, k, i, _RECT, frame)

    def cvm_dilate_cross(self, name: str, k=3, i=1, frame: Optional[NDArray] = None):
        return self._cvm_dilate(name, k, i, _CROSS, frame)

    def cvm_dilate_ellipse(self, name: str, k=3, i=1, frame: Optional[NDArray] = None):
        return self._cvm_dilate(name, k, i, _ELLIPSE, frame)


class CvmMorphologyEx(LayerManagerMixinBase):
    def _cvm_morphology_ex(
        self,
        name: str,
        k: int,
        i: int,
        shape: MorphShape,
        op: MorphOperator,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            layer.param(_SHAPE_PARAMETER_KEY).build_enum(shape)
            m = layer.param("m").build_uint(k, 1, cacher=partial(_gse, layer)).cache
            i = layer.param("i").build_uint(i, 1).value
            o = layer.param("op").build_enum(op).value
            ax = layer.param("anchor_x").build_int(-1).value
            ay = layer.param("anchor_y").build_int(-1).value
            src = frame if frame is not None else layer.prev_frame
            result = morphology_ex(src, o, m, (ax, ay), i)
            layer.frame = result
        return result

    def cvm_morphology_ex_rect_erode(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _RECT, _ERODE, frame)

    def cvm_morphology_ex_rect_dilate(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _RECT, _DILATE, frame)

    def cvm_morphology_ex_rect_open(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _RECT, _OPEN, frame)

    def cvm_morphology_ex_rect_close(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _RECT, _CLOSE, frame)

    def cvm_morphology_ex_rect_gradient(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _RECT, _GRADIENT, frame)

    def cvm_morphology_ex_rect_tophat(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _RECT, _TOPHAT, frame)

    def cvm_morphology_ex_rect_blackhat(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _RECT, _BLACKHAT, frame)

    def cvm_morphology_ex_rect_hitmiss(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _RECT, _HITMISS, frame)

    def cvm_morphology_ex_cross_erode(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _CROSS, _ERODE, frame)

    def cvm_morphology_ex_cross_dilate(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _CROSS, _DILATE, frame)

    def cvm_morphology_ex_cross_open(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _CROSS, _OPEN, frame)

    def cvm_morphology_ex_cross_close(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _CROSS, _CLOSE, frame)

    def cvm_morphology_ex_cross_gradient(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _CROSS, _GRADIENT, frame)

    def cvm_morphology_ex_cross_tophat(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _CROSS, _TOPHAT, frame)

    def cvm_morphology_ex_cross_blackhat(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _CROSS, _BLACKHAT, frame)

    def cvm_morphology_ex_cross_hitmiss(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _CROSS, _HITMISS, frame)

    def cvm_morphology_ex_ellipse_erode(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _ELLIPSE, _ERODE, frame)

    def cvm_morphology_ex_ellipse_dilate(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _ELLIPSE, _DILATE, frame)

    def cvm_morphology_ex_ellipse_open(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _ELLIPSE, _OPEN, frame)

    def cvm_morphology_ex_ellipse_close(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _ELLIPSE, _CLOSE, frame)

    def cvm_morphology_ex_ellipse_gradient(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _ELLIPSE, _GRADIENT, frame)

    def cvm_morphology_ex_ellipse_tophat(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _ELLIPSE, _TOPHAT, frame)

    def cvm_morphology_ex_ellipse_blackhat(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _ELLIPSE, _BLACKHAT, frame)

    def cvm_morphology_ex_ellipse_hitmiss(
        self, name: str, k=3, i=1, frame: Optional[NDArray] = None
    ):
        return self._cvm_morphology_ex(name, k, i, _ELLIPSE, _HITMISS, frame)


class CvmMorphology(
    CvmMorphologyErode,
    CvmMorphologyDilate,
    CvmMorphologyEx,
):
    pass

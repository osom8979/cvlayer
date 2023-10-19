# -*- coding: utf-8 -*-

from functools import partial

from cvlayer.cv.morphology import (
    MorphOperator,
    MorphShape,
    get_structuring_element,
    morphology_ex,
)
from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase

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


def _m(shape: MorphShape, _old, _new):
    return get_structuring_element(shape, (_new, _new))


class CvmMorphology(_LayerManagerMixinBase):
    def _cvm_morphology_ex(
        self,
        name: str,
        shape: MorphShape,
        op: MorphOperator,
        k: int,
        i: int,
    ):
        with self.layer(name) as layer:
            m = layer.param("m").build_unsigned(k, 1, cacher=partial(_m, shape)).cache
            i = layer.param("i").build_unsigned(i, 1).value
            result = morphology_ex(layer.prev_frame, op, m, iterations=i)
            layer.frame = result
        return result

    def cvm_rect_morphology_ex_erode(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _RECT, _ERODE, k, i)

    def cvm_rect_morphology_ex_dilate(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _RECT, _DILATE, k, i)

    def cvm_rect_morphology_ex_open(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _RECT, _OPEN, k, i)

    def cvm_rect_morphology_ex_close(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _RECT, _CLOSE, k, i)

    def cvm_rect_morphology_ex_gradient(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _RECT, _GRADIENT, k, i)

    def cvm_rect_morphology_ex_tophat(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _RECT, _TOPHAT, k, i)

    def cvm_rect_morphology_ex_blackhat(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _RECT, _BLACKHAT, k, i)

    def cvm_rect_morphology_ex_hitmiss(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _RECT, _HITMISS, k, i)

    def cvm_cross_morphology_ex_erode(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _CROSS, _ERODE, k, i)

    def cvm_cross_morphology_ex_dilate(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _CROSS, _DILATE, k, i)

    def cvm_cross_morphology_ex_open(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _CROSS, _OPEN, k, i)

    def cvm_cross_morphology_ex_close(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _CROSS, _CLOSE, k, i)

    def cvm_cross_morphology_ex_gradient(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _CROSS, _GRADIENT, k, i)

    def cvm_cross_morphology_ex_tophat(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _CROSS, _TOPHAT, k, i)

    def cvm_cross_morphology_ex_blackhat(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _CROSS, _BLACKHAT, k, i)

    def cvm_cross_morphology_ex_hitmiss(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _CROSS, _HITMISS, k, i)

    def cvm_ellipse_morphology_ex_erode(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _ELLIPSE, _ERODE, k, i)

    def cvm_ellipse_morphology_ex_dilate(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _ELLIPSE, _DILATE, k, i)

    def cvm_ellipse_morphology_ex_open(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _ELLIPSE, _OPEN, k, i)

    def cvm_ellipse_morphology_ex_close(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _ELLIPSE, _CLOSE, k, i)

    def cvm_ellipse_morphology_ex_gradient(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _ELLIPSE, _GRADIENT, k, i)

    def cvm_ellipse_morphology_ex_tophat(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _ELLIPSE, _TOPHAT, k, i)

    def cvm_ellipse_morphology_ex_blackhat(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _ELLIPSE, _BLACKHAT, k, i)

    def cvm_ellipse_morphology_ex_hitmiss(self, name: str, k=3, i=1):
        return self._cvm_morphology_ex(name, _ELLIPSE, _HITMISS, k, i)

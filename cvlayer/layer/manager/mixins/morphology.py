# -*- coding: utf-8 -*-

from functools import partial

from cvlayer.cv.morphology import (
    MorphOperator,
    MorphShape,
    get_structuring_element,
    morphology_ex,
)
from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase


def _m(shape: MorphShape, _old, _new):
    return get_structuring_element(shape, (_new, _new))


class CvmMorphology(_LayerManagerMixinBase):
    def _cvm_morphology_ex(self, shape: MorphShape, op: MorphOperator, k: int, i: int):
        shape_name = shape.name.lower()
        operator_name = op.name.lower()
        layer_name = f"cvm_{shape_name}_morphology_ex_{operator_name}"
        with self.layer(layer_name) as layer:
            m = layer.param("m").build_unsigned(k, 1, cacher=partial(_m, shape)).cache
            i = layer.param("i").build_unsigned(i, 1).value
            result = morphology_ex(layer.prev_frame, op, m, iterations=i)
            layer.frame = result
        return result

    def cvm_rect_morphology_ex_erode(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.RECT, MorphOperator.ERODE, k, i)

    def cvm_rect_morphology_ex_dilate(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.RECT, MorphOperator.DILATE, k, i)

    def cvm_rect_morphology_ex_open(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.RECT, MorphOperator.OPEN, k, i)

    def cvm_rect_morphology_ex_close(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.RECT, MorphOperator.CLOSE, k, i)

    def cvm_rect_morphology_ex_gradient(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.RECT, MorphOperator.GRADIENT, k, i)

    def cvm_rect_morphology_ex_tophat(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.RECT, MorphOperator.TOPHAT, k, i)

    def cvm_rect_morphology_ex_blackhat(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.RECT, MorphOperator.BLACKHAT, k, i)

    def cvm_rect_morphology_ex_hitmiss(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.RECT, MorphOperator.HITMISS, k, i)

    def cvm_cross_morphology_ex_erode(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.CROSS, MorphOperator.ERODE, k, i)

    def cvm_cross_morphology_ex_dilate(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.CROSS, MorphOperator.DILATE, k, i)

    def cvm_cross_morphology_ex_open(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.CROSS, MorphOperator.OPEN, k, i)

    def cvm_cross_morphology_ex_close(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.CROSS, MorphOperator.CLOSE, k, i)

    def cvm_cross_morphology_ex_gradient(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.CROSS, MorphOperator.GRADIENT, k, i)

    def cvm_cross_morphology_ex_tophat(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.CROSS, MorphOperator.TOPHAT, k, i)

    def cvm_cross_morphology_ex_blackhat(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.CROSS, MorphOperator.BLACKHAT, k, i)

    def cvm_cross_morphology_ex_hitmiss(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.CROSS, MorphOperator.HITMISS, k, i)

    def cvm_ellipse_morphology_ex_erode(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.ELLIPSE, MorphOperator.ERODE, k, i)

    def cvm_ellipse_morphology_ex_dilate(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.ELLIPSE, MorphOperator.DILATE, k, i)

    def cvm_ellipse_morphology_ex_open(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.ELLIPSE, MorphOperator.OPEN, k, i)

    def cvm_ellipse_morphology_ex_close(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.ELLIPSE, MorphOperator.CLOSE, k, i)

    def cvm_ellipse_morphology_ex_gradient(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.ELLIPSE, MorphOperator.GRADIENT, k, i)

    def cvm_ellipse_morphology_ex_tophat(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.ELLIPSE, MorphOperator.TOPHAT, k, i)

    def cvm_ellipse_morphology_ex_blackhat(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.ELLIPSE, MorphOperator.BLACKHAT, k, i)

    def cvm_ellipse_morphology_ex_hitmiss(self, k=3, i=1):
        return self._cvm_morphology_ex(MorphShape.ELLIPSE, MorphOperator.HITMISS, k, i)

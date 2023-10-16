# -*- coding: utf-8 -*-

from cvlayer.cv.morphology import (
    get_morph_rect,
    morphology_ex_close,
    morphology_ex_gradient,
    morphology_ex_open,
)
from cvlayer.layer.mixins._base import _LayerManagerMixinBase


class CvmRectMorphologyEx(_LayerManagerMixinBase):
    @staticmethod
    def _m_rect_cacher(_old, _new):
        return get_morph_rect((_new, _new))

    def cvm_rect_morphology_ex_open(self, k=3, i=1):
        with self._layer("cvm_rect_morphology_ex_open") as layer:
            m = layer.param("m").build_unsigned(k, 1, cacher=self._m_rect_cacher).cache
            i = layer.param("i").build_unsigned(i, 1).value
            result = morphology_ex_open(layer.prev_frame, m, iterations=i)
            layer.frame = result
        return result

    def cvm_rect_morphology_ex_close(self, k=3, i=1):
        with self._layer("cvm_rect_morphology_ex_close") as layer:
            m = layer.param("m").build_unsigned(k, 1, cacher=self._m_rect_cacher).cache
            i = layer.param("i").build_unsigned(i, 1).value
            result = morphology_ex_close(layer.prev_frame, m, iterations=i)
            layer.frame = result
        return result

    def cvm_rect_morphology_ex_gradient(self, k=3, i=1):
        with self._layer("cvm_rect_morphology_ex_gradient") as layer:
            m = layer.param("m").build_unsigned(k, 1, cacher=self._m_rect_cacher).cache
            i = layer.param("i").build_unsigned(i, 1).value
            result = morphology_ex_gradient(layer.prev_frame, m, iterations=i)
            layer.frame = result
        return result

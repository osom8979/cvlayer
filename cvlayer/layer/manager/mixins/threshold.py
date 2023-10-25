# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.threshold import ADAPTIVE_THRESHOLD_METHOD_EXCLUDES as _ATM_EXCLUDES
from cvlayer.cv.threshold import DEFAULT_BLOCK_SIZE as _BS
from cvlayer.cv.threshold import DEFAULT_C as _C
from cvlayer.cv.threshold import PIXEL_8BIT_HALF as _PIX_HALF
from cvlayer.cv.threshold import PIXEL_8BIT_MAX as _PIX_MAX
from cvlayer.cv.threshold import (
    AdaptiveMethod,
    ThresholdMethod,
    adaptive_threshold,
    threshold,
    threshold_otsu,
    threshold_triangle,
)
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase

_MEAN = AdaptiveMethod.MEAN
_GAUSSIAN = AdaptiveMethod.GAUSSIAN

_BINARY = ThresholdMethod.BINARY
_BINARY_INV = ThresholdMethod.BINARY_INV
_TRUNC = ThresholdMethod.TRUNC
_TOZERO = ThresholdMethod.TOZERO
_TOZERO_INV = ThresholdMethod.TOZERO_INV


class CvmThreshold(LayerManagerMixinBase):
    def _cvm_threshold(
        self,
        name: str,
        thresh: int,
        max_value: int,
        method: ThresholdMethod,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            t = layer.param("thresh").build_uint(thresh).value
            mv = layer.param("max").build_uint(max_value).value
            m = layer.param("method").build_enum(method).value
            src = frame if frame is not None else layer.prev_frame
            result = threshold(src, t, mv, m).threshold_image
            layer.frame = result
        return result

    def cvm_threshold_binary(
        self, name: str, t=_PIX_HALF, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold(name, t, mv, _BINARY, frame)

    def cvm_threshold_binary_inv(
        self, name: str, t=_PIX_HALF, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold(name, t, mv, _BINARY_INV, frame)

    def cvm_threshold_trunc(
        self, name: str, t=_PIX_HALF, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold(name, t, mv, _TRUNC, frame)

    def cvm_threshold_tozero(
        self, name: str, t=_PIX_HALF, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold(name, t, mv, _TOZERO, frame)

    def cvm_threshold_tozero_inv(
        self, name: str, t=_PIX_HALF, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold(name, t, mv, _TOZERO_INV, frame)

    def _cvm_threshold_otsu(
        self,
        name: str,
        max_value: int,
        method: ThresholdMethod,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            mv = layer.param("max").build_uint(max_value).value
            m = layer.param("method").build_enum(method).value
            src = frame if frame is not None else layer.prev_frame
            result = threshold_otsu(src, mv, m)
            threshold_value = result.computed_threshold_value
            threshold_image = result.threshold_image
            layer.param("thresh").build_readonly(0.0).value = threshold_value
            layer.frame = threshold_image
        return threshold_image

    def cvm_threshold_otsu_binary(
        self, name: str, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold_otsu(name, mv, _BINARY, frame)

    def cvm_threshold_otsu_binary_inv(
        self, name: str, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold_otsu(name, mv, _BINARY_INV, frame)

    def cvm_threshold_otsu_trunc(
        self, name: str, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold_otsu(name, mv, _TRUNC, frame)

    def cvm_threshold_otsu_tozero(
        self, name: str, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold_otsu(name, mv, _TOZERO, frame)

    def cvm_threshold_otsu_tozero_inv(
        self, name: str, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold_otsu(name, mv, _TOZERO_INV, frame)

    def _cvm_threshold_triangle(
        self,
        name: str,
        max_value: int,
        method: ThresholdMethod,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            mv = layer.param("max").build_uint(max_value).value
            m = layer.param("method").build_enum(method).value
            src = frame if frame is not None else layer.prev_frame
            result = threshold_triangle(src, mv, m)
            threshold_value = result.computed_threshold_value
            threshold_image = result.threshold_image
            layer.param("thresh").build_readonly(0.0).value = threshold_value
            layer.frame = threshold_image
        return threshold_image

    def cvm_threshold_triangle_binary(
        self, name: str, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold_triangle(name, mv, _BINARY, frame)

    def cvm_threshold_triangle_binary_inv(
        self, name: str, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold_triangle(name, mv, _BINARY_INV, frame)

    def cvm_threshold_triangle_trunc(
        self, name: str, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold_triangle(name, mv, _TRUNC, frame)

    def cvm_threshold_triangle_tozero(
        self, name: str, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold_triangle(name, mv, _TOZERO, frame)

    def cvm_threshold_triangle_tozero_inv(
        self, name: str, mv=_PIX_MAX, frame: Optional[NDArray] = None
    ):
        return self._cvm_threshold_triangle(name, mv, _TOZERO_INV, frame)

    def _cvm_adaptive_threshold(
        self,
        name: str,
        max_value: int,
        block_size: int,
        constant: int,
        adaptive_method: AdaptiveMethod,
        method: ThresholdMethod,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            mv = layer.param("max").build_uint(max_value).value
            bs = layer.param("block").build_uint(block_size, min_value=3, step=2).value
            c = layer.param("c").build_int(constant).value
            a = layer.param("adaptive").build_enum(adaptive_method).value
            m = layer.param("method").build_enum(method, excludes=_ATM_EXCLUDES).value
            src = frame if frame is not None else layer.prev_frame
            result = adaptive_threshold(src, mv, a, m, bs, c)
            layer.frame = result
        return result

    def cvm_adaptive_threshold_mean_binary(
        self, name: str, mv=_PIX_MAX, bs=_BS, c=_C, frame: Optional[NDArray] = None
    ):
        return self._cvm_adaptive_threshold(name, mv, bs, c, _MEAN, _BINARY, frame)

    def cvm_adaptive_threshold_mean_binary_inv(
        self, name: str, mv=_PIX_MAX, bs=_BS, c=_C, frame: Optional[NDArray] = None
    ):
        return self._cvm_adaptive_threshold(name, mv, bs, c, _MEAN, _BINARY_INV, frame)

    def cvm_adaptive_threshold_gaussian_binary(
        self, name: str, mv=_PIX_MAX, bs=_BS, c=_C, frame: Optional[NDArray] = None
    ):
        return self._cvm_adaptive_threshold(name, mv, bs, c, _GAUSSIAN, _BINARY, frame)

    def cvm_adaptive_threshold_gaussian_binary_inv(
        self, name: str, mv=_PIX_MAX, bs=_BS, c=_C, frame: Optional[NDArray] = None
    ):
        return self._cvm_adaptive_threshold(
            name, mv, bs, c, _GAUSSIAN, _BINARY_INV, frame
        )

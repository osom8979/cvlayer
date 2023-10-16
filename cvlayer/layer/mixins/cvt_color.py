# -*- coding: utf-8 -*-

from cvlayer.cv.cvt_color import (
    cvt_color_BGR2GRAY,
    cvt_color_BGR2HLS,
    cvt_color_BGR2HSV,
    cvt_color_BGR2LAB,
    cvt_color_BGR2YCR_CB,
    cvt_color_BGR2YUV,
)
from cvlayer.layer.mixins._base import _LayerManagerMixinBase


class CvmCvtColor(_LayerManagerMixinBase):
    def cvm_cvt_color_bgr2gray(self):
        with self._layer("cvm_cvt_color_bgr2gray") as layer:
            layer.frame = gray = cvt_color_BGR2GRAY(layer.prev_frame)
        return gray

    def cvm_cvt_color_bgr2hls(self):
        with self._layer("cvm_cvt_color_bgr2hls") as layer:
            layer.frame = hls = cvt_color_BGR2HLS(layer.prev_frame)
        with self._layer("cvm_cvt_color_bgr2hls_h") as layer:
            layer.frame = hls_h = hls[:, :, 0]
        with self._layer("cvm_cvt_color_bgr2hls_l") as layer:
            layer.frame = hls_l = hls[:, :, 1]
        with self._layer("cvm_cvt_color_bgr2hls_s") as layer:
            layer.frame = hls_s = hls[:, :, 2]
        return hls, hls_h, hls_l, hls_s

    def cvm_cvt_color_bgr2hsv(self):
        with self._layer("cvm_cvt_color_bgr2hsv") as layer:
            layer.frame = hsv = cvt_color_BGR2HSV(layer.prev_frame)
        with self._layer("cvm_cvt_color_bgr2hsv_h") as layer:
            layer.frame = hsv_h = hsv[:, :, 0]
        with self._layer("cvm_cvt_color_bgr2hsv_l") as layer:
            layer.frame = hsv_s = hsv[:, :, 1]
        with self._layer("cvm_cvt_color_bgr2hsv_s") as layer:
            layer.frame = hsv_v = hsv[:, :, 2]
        return hsv, hsv_h, hsv_s, hsv_v

    def cvm_cvt_color_bgr2yuv(self):
        with self._layer("cvm_cvt_color_bgr2yuv") as layer:
            layer.frame = yuv = cvt_color_BGR2YUV(layer.prev_frame)
        with self._layer("cvm_cvt_color_bgr2yuv_h") as layer:
            layer.frame = yuv_y = yuv[:, :, 0]
        with self._layer("cvm_cvt_color_bgr2yuv_l") as layer:
            layer.frame = yuv_u = yuv[:, :, 1]
        with self._layer("cvm_cvt_color_bgr2yuv_s") as layer:
            layer.frame = yuv_v = yuv[:, :, 2]
        return yuv, yuv_y, yuv_u, yuv_v

    def cvm_cvt_color_bgr2ycrcb(self):
        with self._layer("cvm_cvt_color_bgr2ycrcb") as layer:
            layer.frame = ycrcb = cvt_color_BGR2YCR_CB(layer.prev_frame)
        with self._layer("cvm_cvt_color_bgr2ycrcb_y") as layer:
            layer.frame = ycrcb_y = ycrcb[:, :, 0]
        with self._layer("cvm_cvt_color_bgr2ycrcb_cr") as layer:
            layer.frame = ycrcb_cr = ycrcb[:, :, 1]
        with self._layer("cvm_cvt_color_bgr2ycrcb_cb") as layer:
            layer.frame = ycrcb_cb = ycrcb[:, :, 2]
        return ycrcb, ycrcb_y, ycrcb_cr, ycrcb_cb

    def cvm_cvt_color_bgr2lab(self):
        with self._layer("cvm_cvt_color_bgr2lab") as layer:
            layer.frame = lab = cvt_color_BGR2LAB(layer.prev_frame)
        with self._layer("cvm_cvt_color_bgr2lab_l") as layer:
            layer.frame = lab_l = lab[:, :, 0]
        with self._layer("cvm_cvt_color_bgr2lab_a") as layer:
            layer.frame = lab_a = lab[:, :, 1]
        with self._layer("cvm_cvt_color_bgr2lab_b") as layer:
            layer.frame = lab_b = lab[:, :, 2]
        return lab, lab_l, lab_a, lab_b

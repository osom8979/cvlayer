# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.basic import merge, split
from cvlayer.cv.color import shift_degree_channel
from cvlayer.cv.cvt_color import (
    cvt_color_BGR2GRAY,
    cvt_color_BGR2HLS,
    cvt_color_BGR2HSV,
    cvt_color_BGR2LAB,
    cvt_color_BGR2YCR_CB,
    cvt_color_BGR2YUV,
)
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmCvtColor(LayerManagerMixinBase):
    def cvm_cvt_color_bgr2gray(self, name: str, frame: Optional[NDArray] = None):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            layer.frame = gray = cvt_color_BGR2GRAY(src)
        return gray

    def cvm_cvt_color_bgr2hls(self, name: str, frame: Optional[NDArray] = None):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            layer.frame = hls = cvt_color_BGR2HLS(src)
        return hls

    def cvm_cvt_color_bgr2hsv(self, name: str, frame: Optional[NDArray] = None):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            layer.frame = hsv = cvt_color_BGR2HSV(src)
        return hsv

    def cvm_cvt_color_bgr2yuv(self, name: str, frame: Optional[NDArray] = None):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            layer.frame = yuv = cvt_color_BGR2YUV(src)
        return yuv

    def cvm_cvt_color_bgr2ycrcb(self, name: str, frame: Optional[NDArray] = None):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            layer.frame = ycrcb = cvt_color_BGR2YCR_CB(src)
        return ycrcb

    def cvm_cvt_color_bgr2lab(self, name: str, frame: Optional[NDArray] = None):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            layer.frame = lab = cvt_color_BGR2LAB(src)
        return lab

    def cvm_cvt_color_bgr2hsv_hshift(
        self,
        name: str,
        h_shift=0,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            _hs = layer.param("hshift").build_int(h_shift).value
            prev_hsv = cvt_color_BGR2HSV(src)
            prev_h, s, v = split(prev_hsv)
            post_h = shift_degree_channel(prev_h, _hs)
            post_hsv = merge([post_h, s, v])
            layer.frame = post_hsv
        return post_hsv, post_h, s, v

# -*- coding: utf-8 -*-

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
from cvlayer.layer.manager.mixins._base import _LayerManagerMixinBase


class CvmCvtColor(_LayerManagerMixinBase):
    def cvm_cvt_color_bgr2gray(self, name: str):
        with self.layer(name) as layer:
            layer.frame = gray = cvt_color_BGR2GRAY(layer.prev_frame)
        return gray

    def cvm_cvt_color_bgr2hls(self, name: str):
        with self.layer(name) as layer:
            layer.frame = hls = cvt_color_BGR2HLS(layer.prev_frame)
        return hls

    def cvm_cvt_color_bgr2hsv(self, name: str):
        with self.layer(name) as layer:
            layer.frame = hsv = cvt_color_BGR2HSV(layer.prev_frame)
        return hsv

    def cvm_cvt_color_bgr2yuv(self, name: str):
        with self.layer(name) as layer:
            layer.frame = yuv = cvt_color_BGR2YUV(layer.prev_frame)
        return yuv

    def cvm_cvt_color_bgr2ycrcb(self, name: str):
        with self.layer(name) as layer:
            layer.frame = ycrcb = cvt_color_BGR2YCR_CB(layer.prev_frame)
        return ycrcb

    def cvm_cvt_color_bgr2lab(self, name: str):
        with self.layer(name) as layer:
            layer.frame = lab = cvt_color_BGR2LAB(layer.prev_frame)
        return lab

    def cvm_cvt_color_bgr2hsv_hshift(self, name: str):
        with self.layer(name) as layer:
            hshift = layer.param("hshift").build_unsigned(20, 1).value
            prev_hsv = cvt_color_BGR2HSV(layer.prev_frame)
            prev_h, s, v = split(prev_hsv)
            post_h = shift_degree_channel(prev_h, hshift)
            post_hsv = merge([post_h, s, v])
            layer.frame = post_hsv
        return post_hsv

# -*- coding: utf-8 -*-

from math import pi
from typing import Optional

from numpy import cos, sin
from numpy.typing import NDArray

from cvlayer.cv.drawable.line import draw_line_coord
from cvlayer.cv.hough_lines import (
    DEFAULT_RHO,
    DEFAULT_THETA,
    DEFAULT_THRESHOLD,
    hough_lines,
)
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class CvmHoughLines(LayerManagerMixinBase):
    def cvm_hough_lines(
        self,
        name: str,
        rho=DEFAULT_RHO,
        theta=DEFAULT_THETA,
        threshold=DEFAULT_THRESHOLD,
        srn=0.0,
        stn=0.0,
        min_theta=0.0,
        max_theta=pi,
        canvas: Optional[NDArray] = None,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            _rho = layer.param("rho").build_float(rho).value
            _theta = layer.param("theta").build_float(theta).value
            _threshold = layer.param("threshold").build_uint(threshold).value
            _srn = layer.param("srn").build_float(srn).value
            _stn = layer.param("stn").build_float(stn).value
            _min_theta = layer.param("min_theta").build_float(min_theta).value
            _max_theta = layer.param("max_theta").build_float(max_theta).value

            src = frame if frame is not None else layer.prev_frame
            canvas = canvas if canvas is not None else src.copy()

            lines = hough_lines(
                src,
                _rho,
                _theta,
                _threshold,
                _srn,
                _stn,
                _min_theta,
                _max_theta,
            )

            if lines is not None:
                for rho, theta in lines[:, 0].tolist():
                    a = cos(theta)
                    b = sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)
                    draw_line_coord(canvas, x1, y1, x2, y2, (0, 0, 255), 2)

            layer.frame = canvas

        return lines

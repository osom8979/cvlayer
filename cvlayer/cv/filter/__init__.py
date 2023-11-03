# -*- coding: utf-8 -*-

from cvlayer.cv.filter.blur import CvlFilterBlur
from cvlayer.cv.filter.d2 import CvlFilterD2
from cvlayer.cv.filter.denoising import CvlFilterDenoising
from cvlayer.cv.filter.edge import CvlFilterEdge


class CvlFilter(
    CvlFilterBlur,
    CvlFilterD2,
    CvlFilterDenoising,
    CvlFilterEdge,
):
    pass

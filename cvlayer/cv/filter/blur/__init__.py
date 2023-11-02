# -*- coding: utf-8 -*-

from cvlayer.cv.filter.blur.bilateral import CvlFilterBlurBilateral
from cvlayer.cv.filter.blur.blur import CvlFilterBlurBlur
from cvlayer.cv.filter.blur.gaussian import CvlFilterBlurGaussian
from cvlayer.cv.filter.blur.median import CvlFilterBlurMedian


class CvlFilterBlur(
    CvlFilterBlurBilateral,
    CvlFilterBlurBlur,
    CvlFilterBlurGaussian,
    CvlFilterBlurMedian,
):
    pass

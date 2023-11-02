# -*- coding: utf-8 -*-

from cvlayer.layer.manager.mixins.filter.blur.bilateral import CvmFilterBlurBilateral
from cvlayer.layer.manager.mixins.filter.blur.gaussian import CvmFilterBlurGaussian


class CvmFilterBlur(
    CvmFilterBlurBilateral,
    CvmFilterBlurGaussian,
):
    pass

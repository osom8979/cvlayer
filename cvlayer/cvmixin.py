# -*- coding: utf-8 -*-

from cvlayer.layer.manager.mixins.blur import CvmBlur
from cvlayer.layer.manager.mixins.canny import CvmCanny
from cvlayer.layer.manager.mixins.cvt_color import CvmCvtColor
from cvlayer.layer.manager.mixins.morphology import CvmMorphology


class CvMixin(
    CvmBlur,
    CvmCanny,
    CvmCvtColor,
    CvmMorphology,
):
    pass

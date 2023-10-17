# -*- coding: utf-8 -*-

from cvlayer.layer.manager.mixins.blur import CvmBlur
from cvlayer.layer.manager.mixins.canny import CvmCanny
from cvlayer.layer.manager.mixins.cvt_color import CvmCvtColor
from cvlayer.layer.manager.mixins.histogram import CvmHistogram
from cvlayer.layer.manager.mixins.kmeans import CvmKmeans
from cvlayer.layer.manager.mixins.morphology import CvmMorphology
from cvlayer.layer.manager.mixins.select import CvmSelect
from cvlayer.layer.manager.mixins.threshold import CvmThreshold


class CvMixin(
    CvmBlur,
    CvmCanny,
    CvmCvtColor,
    CvmHistogram,
    CvmKmeans,
    CvmMorphology,
    CvmSelect,
    CvmThreshold,
):
    pass

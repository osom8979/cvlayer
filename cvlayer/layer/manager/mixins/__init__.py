# -*- coding: utf-8 -*-

from cvlayer.layer.manager.mixins.basic import CvmBasic
from cvlayer.layer.manager.mixins.bitwise import CvmBitwise
from cvlayer.layer.manager.mixins.border import CvmBorder
from cvlayer.layer.manager.mixins.contours import CvmContours
from cvlayer.layer.manager.mixins.cvt_color import CvmCvtColor
from cvlayer.layer.manager.mixins.filter import CvmFilter
from cvlayer.layer.manager.mixins.histogram import CvmHistogram
from cvlayer.layer.manager.mixins.kmeans import CvmKmeans
from cvlayer.layer.manager.mixins.mean_std_dev import CvmMeanStdDev
from cvlayer.layer.manager.mixins.morphology import CvmMorphology
from cvlayer.layer.manager.mixins.pyramid import CvmPyramid
from cvlayer.layer.manager.mixins.threshold import CvmThreshold
from cvlayer.layer.manager.mixins.transform import CvmTransform
from cvlayer.layer.manager.mixins.utils import CvmUtils


class CvMixin(
    CvmBasic,
    CvmBitwise,
    CvmBorder,
    CvmFilter,
    CvmContours,
    CvmCvtColor,
    CvmHistogram,
    CvmKmeans,
    CvmMeanStdDev,
    CvmMorphology,
    CvmPyramid,
    CvmThreshold,
    CvmTransform,
    CvmUtils,
):
    pass

# -*- coding: utf-8 -*-

from cvlayer.layer.manager.mixins.filter.blur import CvmFilterBlur
from cvlayer.layer.manager.mixins.filter.edge import CvmFilterEdge


class CvmFilter(
    CvmFilterBlur,
    CvmFilterEdge,
):
    pass

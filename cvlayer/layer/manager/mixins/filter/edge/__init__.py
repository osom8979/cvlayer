# -*- coding: utf-8 -*-

from cvlayer.layer.manager.mixins.filter.edge.canny import CvmFilterEdgeCanny
from cvlayer.layer.manager.mixins.filter.edge.laplacian import CvmFilterEdgeLaplacian


class CvmFilterEdge(
    CvmFilterEdgeCanny,
    CvmFilterEdgeLaplacian,
):
    pass

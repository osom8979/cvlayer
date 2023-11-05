# -*- coding: utf-8 -*-

from cvlayer.layer.manager.mixins.filter.edge.canny import CvmFilterEdgeCanny
from cvlayer.layer.manager.mixins.filter.edge.laplacian import CvmFilterEdgeLaplacian
from cvlayer.layer.manager.mixins.filter.edge.sobel_magnitude import (
    CvmFilterEdgeSobelMagnitude,
)


class CvmFilterEdge(
    CvmFilterEdgeCanny,
    CvmFilterEdgeLaplacian,
    CvmFilterEdgeSobelMagnitude,
):
    pass

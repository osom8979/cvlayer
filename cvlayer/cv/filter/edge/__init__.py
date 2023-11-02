# -*- coding: utf-8 -*-

from cvlayer.cv.filter.edge.canny import CvlFilterEdgeCanny
from cvlayer.cv.filter.edge.laplacian import CvlFilterEdgeLaplacian
from cvlayer.cv.filter.edge.scharr import CvlFilterEdgeScharr
from cvlayer.cv.filter.edge.sobel import CvlFilterEdgeSobel


class CvlFilterEdge(
    CvlFilterEdgeCanny,
    CvlFilterEdgeLaplacian,
    CvlFilterEdgeScharr,
    CvlFilterEdgeSobel,
):
    pass

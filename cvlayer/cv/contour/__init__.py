# -*- coding: utf-8 -*-

from cvlayer.cv.contour.contour import CvlContourContour
from cvlayer.cv.contour.edge import CvlContourEdge
from cvlayer.cv.contour.find import CvlContourFind
from cvlayer.cv.contour.intersection import CvlContourIntersection
from cvlayer.cv.contour.moments import CvlContourMoments


class CvlContour(
    CvlContourContour,
    CvlContourEdge,
    CvlContourFind,
    CvlContourIntersection,
    CvlContourMoments,
):
    pass

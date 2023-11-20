# -*- coding: utf-8 -*-

from cvlayer.cv.contour.contour import CvlContourContour
from cvlayer.cv.contour.find import CvlContourFind
from cvlayer.cv.contour.intersection import CvlContourIntersection
from cvlayer.cv.contour.moments import CvlContourMoments
from cvlayer.cv.contour.most_point import CvlContourMostPoint


class CvlContour(
    CvlContourContour,
    CvlContourFind,
    CvlContourIntersection,
    CvlContourMoments,
    CvlContourMostPoint,
):
    pass

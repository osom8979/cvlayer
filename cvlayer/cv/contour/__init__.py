# -*- coding: utf-8 -*-

from cvlayer.cv.contour.analysis import CvlContourAnalysis
from cvlayer.cv.contour.component import CvlContourComponent
from cvlayer.cv.contour.contour import CvlContourContour
from cvlayer.cv.contour.find import CvlContourFind
from cvlayer.cv.contour.moments import CvlContourMoments
from cvlayer.cv.contour.most_point import CvlContourMostPoint


class CvlContour(
    CvlContourAnalysis,
    CvlContourComponent,
    CvlContourContour,
    CvlContourFind,
    CvlContourMoments,
    CvlContourMostPoint,
):
    pass

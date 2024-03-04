# -*- coding: utf-8 -*-

from cvlayer.cv.drawable.arrowed import CvlDrawableArrowed
from cvlayer.cv.drawable.circle import CvlDrawableCircle
from cvlayer.cv.drawable.contours import CvlDrawableContours
from cvlayer.cv.drawable.crosshair import CvlDrawableCrosshair
from cvlayer.cv.drawable.ellipse import CvlDrawableEllipse
from cvlayer.cv.drawable.image import CvlDrawableImage
from cvlayer.cv.drawable.keypoint import CvlDrawableKeyPoints
from cvlayer.cv.drawable.line import CvlDrawableLine
from cvlayer.cv.drawable.marker import CvlDrawableMarker
from cvlayer.cv.drawable.plot import CvlDrawablePlot
from cvlayer.cv.drawable.point import CvlDrawablePoint
from cvlayer.cv.drawable.rectangle import CvlDrawableRectangle
from cvlayer.cv.drawable.text import CvlDrawableText


class CvlDrawable(
    CvlDrawableArrowed,
    CvlDrawableCircle,
    CvlDrawableContours,
    CvlDrawableCrosshair,
    CvlDrawableEllipse,
    CvlDrawableImage,
    CvlDrawableKeyPoints,
    CvlDrawableLine,
    CvlDrawableMarker,
    CvlDrawablePlot,
    CvlDrawablePoint,
    CvlDrawableRectangle,
    CvlDrawableText,
):
    pass

# -*- coding: utf-8 -*-

from cvlayer.cvl.backend import CvlBackend
from cvlayer.cvl.bgsub import CvlBackgroundSubtractor
from cvlayer.cvl.contours import CvlContours
from cvlayer.cvl.contours_edge import CvlContoursEdge
from cvlayer.cvl.cvt_color import CvlCvtColor
from cvlayer.cvl.drawable import CvlDrawable
from cvlayer.cvl.edge_detector import CvlEdgeDetector
from cvlayer.cvl.fourcc import CvlFourcc
from cvlayer.cvl.hsv import CvlHsv
from cvlayer.cvl.image import CvlImage
from cvlayer.cvl.in_range import CvlInRange
from cvlayer.cvl.morphology import CvlMorphology
from cvlayer.cvl.palette import CvlPalette
from cvlayer.cvl.rotate_tracer import CvlRotateTracer
from cvlayer.cvl.threshold import CvlThreshold
from cvlayer.cvl.tracker import CvlTracker


class CvLayer(
    CvlBackend,
    CvlBackgroundSubtractor,
    CvlContours,
    CvlContoursEdge,
    CvlCvtColor,
    CvlDrawable,
    CvlEdgeDetector,
    CvlFourcc,
    CvlHsv,
    CvlImage,
    CvlInRange,
    CvlMorphology,
    CvlPalette,
    CvlRotateTracer,
    CvlThreshold,
    CvlTracker,
):
    pass

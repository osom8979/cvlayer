# -*- coding: utf-8 -*-

from cvlayer.cvl.backend import CvlBackend
from cvlayer.cvl.bgsub import CvlBackgroundSubtractor
from cvlayer.cvl.bitwise import CvlBitwise
from cvlayer.cvl.contours import CvlContours
from cvlayer.cvl.contours_edge import CvlContoursEdge
from cvlayer.cvl.cvt_color import CvlCvtColor
from cvlayer.cvl.drawable import CvlDrawable
from cvlayer.cvl.drawable_contours import CvlDrawableContours
from cvlayer.cvl.edge_detector import CvlEdgeDetector
from cvlayer.cvl.fourcc import CvlFourcc
from cvlayer.cvl.hsv import CvlHsv
from cvlayer.cvl.in_range import CvlInRange
from cvlayer.cvl.make_image import CvlMakeImage
from cvlayer.cvl.morphology import CvlMorphology
from cvlayer.cvl.palette import CvlPalette
from cvlayer.cvl.resize import CvlResize
from cvlayer.cvl.rotate_tracer import CvlRotateTracer
from cvlayer.cvl.threshold import CvlThreshold
from cvlayer.cvl.tracker import CvlTracker
from cvlayer.cvl.video_capture import CvlVideoCapture
from cvlayer.cvl.window import CvlWindow


class CvLayer(
    CvlBackend,
    CvlBackgroundSubtractor,
    CvlBitwise,
    CvlContours,
    CvlContoursEdge,
    CvlCvtColor,
    CvlDrawable,
    CvlDrawableContours,
    CvlEdgeDetector,
    CvlFourcc,
    CvlHsv,
    CvlMakeImage,
    CvlInRange,
    CvlMorphology,
    CvlPalette,
    CvlResize,
    CvlRotateTracer,
    CvlThreshold,
    CvlTracker,
    CvlVideoCapture,
    CvlWindow,
):
    pass

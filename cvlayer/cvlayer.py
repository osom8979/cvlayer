# -*- coding: utf-8 -*-

from cvlayer.cv.backend import CvlBackend
from cvlayer.cv.bgsub import CvlBackgroundSubtractor
from cvlayer.cv.bitwise import CvlBitwise
from cvlayer.cv.contours import CvlContours
from cvlayer.cv.contours_edge import CvlContoursEdge
from cvlayer.cv.contours_intersection import CvlContoursIntersection
from cvlayer.cv.cvt_color import CvlCvtColor
from cvlayer.cv.cvt_shapely import CvlShapely
from cvlayer.cv.drawable import CvlDrawable
from cvlayer.cv.drawable_contours import CvlDrawableContours
from cvlayer.cv.edge_detector import CvlEdgeDetector
from cvlayer.cv.fourcc import CvlFourcc
from cvlayer.cv.hsv import CvlHsv
from cvlayer.cv.in_range import CvlInRange
from cvlayer.cv.intrusion_detection import CvlIntrusionDetection
from cvlayer.cv.keymap import CvlKeymap
from cvlayer.cv.make_image import CvlMakeImage
from cvlayer.cv.morphology import CvlMorphology
from cvlayer.cv.palette import CvlPalette
from cvlayer.cv.perspective import CvlPerspective
from cvlayer.cv.resize import CvlResize
from cvlayer.cv.rotate_tracer import CvlRotateTracer
from cvlayer.cv.threshold import CvlThreshold
from cvlayer.cv.tracker import CvlTracker
from cvlayer.cv.video_capture import CvlVideoCapture
from cvlayer.cv.window import CvlWindow


class CvLayer(
    CvlBackend,
    CvlBackgroundSubtractor,
    CvlBitwise,
    CvlContours,
    CvlContoursEdge,
    CvlContoursIntersection,
    CvlCvtColor,
    CvlShapely,
    CvlDrawable,
    CvlDrawableContours,
    CvlEdgeDetector,
    CvlFourcc,
    CvlHsv,
    CvlMakeImage,
    CvlInRange,
    CvlIntrusionDetection,
    CvlKeymap,
    CvlMorphology,
    CvlPalette,
    CvlPerspective,
    CvlResize,
    CvlRotateTracer,
    CvlThreshold,
    CvlTracker,
    CvlVideoCapture,
    CvlWindow,
):
    pass

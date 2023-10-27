# -*- coding: utf-8 -*-

from cvlayer.cv.transform.distance_transform import CvlTransformDistanceTransform
from cvlayer.cv.transform.flood_fill import CvlTransformFloodFill
from cvlayer.cv.transform.grab_cut import CvlTransformGrabCut
from cvlayer.cv.transform.watershed import CvlTransformWatershed


class CvlTransform(
    CvlTransformDistanceTransform,
    CvlTransformFloodFill,
    CvlTransformGrabCut,
    CvlTransformWatershed,
):
    pass

# -*- coding: utf-8 -*-

from cvlayer.cv.transform.distance_transform import CvlTransformDistanceTransform
from cvlayer.cv.transform.flood_fill import CvlTransformFloodFill


class CvlTransform(
    CvlTransformDistanceTransform,
    CvlTransformFloodFill,
):
    pass

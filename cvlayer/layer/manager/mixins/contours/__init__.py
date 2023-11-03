# -*- coding: utf-8 -*-

from cvlayer.layer.manager.mixins.contours.ex import CvmContoursEx
from cvlayer.layer.manager.mixins.contours.filter_area import CvmContoursFilterArea
from cvlayer.layer.manager.mixins.contours.find import CvmContoursFind
from cvlayer.layer.manager.mixins.contours.hole import CvmContoursHole
from cvlayer.layer.manager.mixins.contours.largest import CvmContoursLargest


class CvmContours(
    CvmContoursEx,
    CvmContoursFilterArea,
    CvmContoursFind,
    CvmContoursHole,
    CvmContoursLargest,
):
    pass

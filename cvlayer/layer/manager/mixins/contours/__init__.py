# -*- coding: utf-8 -*-

from cvlayer.layer.manager.mixins.contours.find import CvmContoursFind
from cvlayer.layer.manager.mixins.contours.punctures import CvmContoursPunctures


class CvmContours(
    CvmContoursFind,
    CvmContoursPunctures,
):
    pass

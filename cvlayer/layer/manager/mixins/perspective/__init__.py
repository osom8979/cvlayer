# -*- coding: utf-8 -*-

from cvlayer.layer.manager.mixins.perspective.select import CvmPerspectiveSelect
from cvlayer.layer.manager.mixins.perspective.transform import CvmPerspectiveTransform


class CvmPerspective(
    CvmPerspectiveSelect,
    CvmPerspectiveTransform,
):
    pass

# -*- coding: utf-8 -*-

from cvlayer.cv.drawable.text.measure import CvlDrawableTextMeasure
from cvlayer.cv.drawable.text.multiline import CvlDrawableTextMultiline
from cvlayer.cv.drawable.text.outline import CvlDrawableTextOutline
from cvlayer.cv.drawable.text.text import CvlDrawableTextText


class CvlDrawableText(
    CvlDrawableTextMeasure,
    CvlDrawableTextMultiline,
    CvlDrawableTextOutline,
    CvlDrawableTextText,
):
    pass

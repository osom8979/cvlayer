# -*- coding: utf-8 -*-

from cvlayer.cv.drawable.text.multiline.box import CvlDrawableTextMultilineBox
from cvlayer.cv.drawable.text.multiline.lines import CvlDrawableTextMultilineLines
from cvlayer.cv.drawable.text.multiline.measure import CvlDrawableTextMultilineMeasure
from cvlayer.cv.drawable.text.multiline.text import CvlDrawableTextMultilineText


class CvlDrawableTextMultiline(
    CvlDrawableTextMultilineBox,
    CvlDrawableTextMultilineLines,
    CvlDrawableTextMultilineMeasure,
    CvlDrawableTextMultilineText,
):
    pass

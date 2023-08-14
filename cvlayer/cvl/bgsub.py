# -*- coding: utf-8 -*-

from typing import Optional

from cvlayer.cv.bgsub import (
    DEFAULT_DETECT_SHADOWS,
    DEFAULT_HISTORY,
    DEFAULT_METHOD,
    BackgroundSubtractor,
    BackgroundSubtractorMethod,
)


class CvlBackgroundSubtractor:
    BackgroundSubtractorMethodType = BackgroundSubtractorMethod

    @staticmethod
    def cvl_create_background_subtractor(
        method=DEFAULT_METHOD,
        history=DEFAULT_HISTORY,
        threshold: Optional[float] = None,
        shadow=DEFAULT_DETECT_SHADOWS,
    ):
        return BackgroundSubtractor(method, history, threshold, shadow)

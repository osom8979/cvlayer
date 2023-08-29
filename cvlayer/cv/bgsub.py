# -*- coding: utf-8 -*-

from enum import Enum, auto, unique
from typing import Any, Final, Optional

import cv2
from numpy import zeros
from numpy.typing import NDArray


@unique
class BackgroundSubtractorMethod(Enum):
    MOG2 = auto()
    KNN = auto()
    # bgsegm_CNT = auto()
    # bgsegm_GMG = auto()
    # bgsegm_GSOC = auto()
    # bgsegm_LSBP = auto()
    # bgsegm_MOG = auto()
    # cuda_FGD = auto()
    # cuda_GMG = auto()
    # cuda_MOG = auto()


DEFAULT_METHOD: Final[BackgroundSubtractorMethod] = BackgroundSubtractorMethod.MOG2
DEFAULT_HISTORY: Final[int] = 500
DEFAULT_MOG2_THRESHOLD: Final[float] = 16.0
DEFAULT_KNN_THRESHOLD: Final[float] = 400.0
DEFAULT_DETECT_SHADOWS: Final[bool] = True


def create_background_subtractor(
    method: BackgroundSubtractorMethod,
    history: int,
    threshold: float,
    shadow: bool,
) -> Any:
    if method == BackgroundSubtractorMethod.MOG2:
        return cv2.createBackgroundSubtractorMOG2(history, threshold, shadow)
    elif method == BackgroundSubtractorMethod.KNN:
        return cv2.createBackgroundSubtractorKNN(history, threshold, shadow)
    else:
        assert False, "Inaccessible section"


def default_threshold(
    method: BackgroundSubtractorMethod,
    value: Optional[float] = None,
) -> float:
    if value is not None:
        return value

    if method == BackgroundSubtractorMethod.MOG2:
        return DEFAULT_MOG2_THRESHOLD
    elif method == BackgroundSubtractorMethod.KNN:
        return DEFAULT_KNN_THRESHOLD
    else:
        assert False, "Inaccessible section"


class BackgroundSubtractor:
    def __init__(
        self,
        method=DEFAULT_METHOD,
        history=DEFAULT_HISTORY,
        threshold: Optional[float] = None,
        shadow=DEFAULT_DETECT_SHADOWS,
    ):
        self._method = method
        self._foreground_mask = zeros(0)
        self._background_subtractor = create_background_subtractor(
            self._method,
            history,
            default_threshold(method, threshold),
            shadow,
        )

        assert hasattr(self._background_subtractor, "apply")
        assert hasattr(self._background_subtractor, "clear")
        assert hasattr(self._background_subtractor, "empty")
        assert hasattr(self._background_subtractor, "getBackgroundImage")
        assert hasattr(self._background_subtractor, "getBackgroundRatio")
        assert hasattr(self._background_subtractor, "getComplexityReductionThreshold")
        assert hasattr(self._background_subtractor, "getDefaultName")
        assert hasattr(self._background_subtractor, "getDetectShadows")
        assert hasattr(self._background_subtractor, "getHistory")
        assert hasattr(self._background_subtractor, "getNMixtures")
        assert hasattr(self._background_subtractor, "getShadowThreshold")
        assert hasattr(self._background_subtractor, "getShadowValue")
        assert hasattr(self._background_subtractor, "getVarInit")
        assert hasattr(self._background_subtractor, "getVarMax")
        assert hasattr(self._background_subtractor, "getVarMin")
        assert hasattr(self._background_subtractor, "getVarThreshold")
        assert hasattr(self._background_subtractor, "getVarThresholdGen")
        assert hasattr(self._background_subtractor, "read")
        assert hasattr(self._background_subtractor, "save")
        assert hasattr(self._background_subtractor, "setBackgroundRatio")
        assert hasattr(self._background_subtractor, "setComplexityReductionThreshold")
        assert hasattr(self._background_subtractor, "setDetectShadows")
        assert hasattr(self._background_subtractor, "setHistory")
        assert hasattr(self._background_subtractor, "setNMixtures")
        assert hasattr(self._background_subtractor, "setShadowThreshold")
        assert hasattr(self._background_subtractor, "setShadowValue")
        assert hasattr(self._background_subtractor, "setVarInit")
        assert hasattr(self._background_subtractor, "setVarMax")
        assert hasattr(self._background_subtractor, "setVarMin")
        assert hasattr(self._background_subtractor, "setVarThreshold")
        assert hasattr(self._background_subtractor, "setVarThresholdGen")
        assert hasattr(self._background_subtractor, "write")
        # print(dir(self._background_subtractor))

    @property
    def method(self) -> BackgroundSubtractorMethod:
        return self._method

    @property
    def history(self) -> int:
        return self._background_subtractor.getHistory()

    @history.setter
    def history(self, value: int) -> None:
        self._background_subtractor.setHistory(value)

    @property
    def threshold(self) -> float:
        return self._background_subtractor.getVarThreshold()

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._background_subtractor.setVarThreshold(value)

    @property
    def shadow(self) -> bool:
        return self._background_subtractor.getShadowValue()

    @shadow.setter
    def shadow(self, value: bool) -> None:
        self._background_subtractor.setShadowValue(value)

    @property
    def background(self) -> NDArray:
        return self._background_subtractor.getBackgroundImage()

    @property
    def foreground_mask(self) -> NDArray:
        return self._foreground_mask

    def apply(self, frame: NDArray) -> NDArray:
        self._foreground_mask = self._background_subtractor.apply(frame)
        assert self._foreground_mask is not None
        return self._foreground_mask


class CvlBackgroundSubtractor:
    @staticmethod
    def cvl_create_background_subtractor(
        method=DEFAULT_METHOD,
        history=DEFAULT_HISTORY,
        threshold: Optional[float] = None,
        shadow=DEFAULT_DETECT_SHADOWS,
    ):
        return BackgroundSubtractor(method, history, threshold, shadow)

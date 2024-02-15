# -*- coding: utf-8 -*-

from enum import Enum, auto, unique
from os import path
from typing import Any, Final, List, Optional

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


DEFAULT_HISTORY: Final[int] = 500
DEFAULT_MOG2_THRESHOLD: Final[float] = 16.0
DEFAULT_KNN_THRESHOLD: Final[float] = 400.0
DEFAULT_DETECT_SHADOWS: Final[bool] = True

BGSUB_CREATE_METHOD_PREFIX: Final[str] = "createBackgroundSubtractor"
BGSUB_CREATE_METHOD_SUFFIX: Final[str] = ""


def find_background_subtractor() -> List[str]:
    return list(filter(lambda x: x.startswith(BGSUB_CREATE_METHOD_PREFIX), dir(cv2)))


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
    _fgmask: NDArray

    def __init__(
        self,
        method: BackgroundSubtractorMethod,
        history=DEFAULT_HISTORY,
        threshold: Optional[float] = None,
        shadow=DEFAULT_DETECT_SHADOWS,
    ):
        self._fgmask = zeros([])
        self._bgsub = create_background_subtractor(
            method, history, default_threshold(method, threshold), shadow
        )

        # Algorithm
        assert hasattr(self._bgsub, "clear")
        assert hasattr(self._bgsub, "empty")
        assert hasattr(self._bgsub, "getDefaultName")
        assert hasattr(self._bgsub, "read")
        assert hasattr(self._bgsub, "save")
        assert hasattr(self._bgsub, "write")

        # BackgroundSubtractor
        assert hasattr(self._bgsub, "apply")
        assert hasattr(self._bgsub, "getBackgroundImage")

    @property
    def foreground_mask(self) -> NDArray:
        """Foreground mask as an 8-bit binary image"""
        return self._fgmask

    def clear(self) -> None:
        """Clears the algorithm state"""
        assert isinstance(self._bgsub, cv2.Algorithm)
        self._bgsub.clear()

    @property
    def empty(self) -> bool:
        """
        Returns true if the Algorithm is empty.
        e.g. in the very beginning or after unsuccessful read.
        """
        assert isinstance(self._bgsub, cv2.Algorithm)
        return self._bgsub.empty()

    @property
    def default_name(self) -> str:
        assert isinstance(self._bgsub, cv2.Algorithm)
        return self._bgsub.getDefaultName()

    def read(self, node: cv2.FileNode) -> None:
        assert isinstance(self._bgsub, cv2.Algorithm)
        self._bgsub.read(node)

    def write(self, storage: cv2.FileStorage, name: Optional[str] = None) -> None:
        assert isinstance(self._bgsub, cv2.Algorithm)
        if name is not None:
            self._bgsub.write(storage, name)
        else:
            self._bgsub.write(storage)

    def save(self, filename: str) -> None:
        assert isinstance(self._bgsub, cv2.Algorithm)
        self._bgsub.save(filename)

    def load(self, filename: str, name: Optional[str] = None) -> None:
        assert isinstance(self._bgsub, cv2.Algorithm)
        if not path.isfile(filename):
            raise FileNotFoundError(f"File not found error: '{filename}'")

        storage = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise EOFError("File is not opened error")

        try:
            node = storage.getNode(name) if name else storage.getFirstTopLevelNode()
            self._bgsub.read(node)
        finally:
            storage.release()

    def apply(self, image: NDArray, learning_rate=-1) -> NDArray:
        """
        Computes a foreground mask.

        :param image: Next video frame.
        :param learning_rate: The value between 0 and 1 that indicates
            how fast the background model is learnt. Negative parameter value makes
            the algorithm to use some automatically chosen learning rate.
            0 means that the background model is not updated at all,
            1 means that the background model is completely reinitialized from the
            last frame.
        """
        assert isinstance(self._bgsub, cv2.BackgroundSubtractor)
        self._fgmask = self._bgsub.apply(image, None, learning_rate)
        return self._fgmask

    @property
    def background(self) -> NDArray:
        """Computes a background image"""
        assert isinstance(self._bgsub, cv2.BackgroundSubtractor)
        return self._bgsub.getBackgroundImage()


class BackgroundSubtractorKNN(BackgroundSubtractor):
    def __init__(
        self,
        history=DEFAULT_HISTORY,
        threshold=DEFAULT_KNN_THRESHOLD,
        shadow=DEFAULT_DETECT_SHADOWS,
    ):
        super().__init__(BackgroundSubtractorMethod.KNN, history, threshold, shadow)

        assert hasattr(self._bgsub, "getDetectShadows")
        assert hasattr(self._bgsub, "getDist2Threshold")
        assert hasattr(self._bgsub, "getHistory")
        assert hasattr(self._bgsub, "getkNNSamples")
        assert hasattr(self._bgsub, "getNSamples")
        assert hasattr(self._bgsub, "getShadowThreshold")
        assert hasattr(self._bgsub, "getShadowValue")

        assert hasattr(self._bgsub, "setDetectShadows")
        assert hasattr(self._bgsub, "setDist2Threshold")
        assert hasattr(self._bgsub, "setHistory")
        assert hasattr(self._bgsub, "setkNNSamples")
        assert hasattr(self._bgsub, "setNSamples")
        assert hasattr(self._bgsub, "setShadowThreshold")
        assert hasattr(self._bgsub, "setShadowValue")

    @property
    def detect_shadows(self) -> bool:
        """Returns the shadow detection flag"""
        return self._bgsub.getDetectShadows()

    @detect_shadows.setter
    def detect_shadows(self, value: bool) -> None:
        self._bgsub.setDetectShadows(value)

    @property
    def dist2_threshold(self) -> float:
        """
        Returns the threshold on the squared distance between the pixel and the sample.
        """
        return self._bgsub.getDist2Threshold()

    @dist2_threshold.setter
    def dist2_threshold(self, value: float) -> None:
        self._bgsub.setDist2Threshold(value)

    @property
    def history(self) -> int:
        """Returns the number of last frames that affect the background model"""
        return self._bgsub.getHistory()

    @history.setter
    def history(self, value: int) -> None:
        self._bgsub.setHistory(value)

    @property
    def knn_samples(self) -> int:
        """Returns the number of neighbours, the k in the kNN"""
        return self._bgsub.getkNNSamples()

    @knn_samples.setter
    def knn_samples(self, value: int) -> None:
        self._bgsub.setkNNSamples(value)

    @property
    def number_samples(self) -> int:
        """Returns the number of data samples in the background model"""
        return self._bgsub.getNSamples()

    @number_samples.setter
    def number_samples(self, value: int) -> None:
        self._bgsub.setNSamples(value)

    @property
    def shadow_threshold(self) -> float:
        """Returns the shadow threshold"""
        return self._bgsub.getShadowThreshold()

    @shadow_threshold.setter
    def shadow_threshold(self, value: float) -> None:
        self._bgsub.setShadowThreshold(value)

    @property
    def shadow_value(self) -> int:
        """Returns the shadow value"""
        return self._bgsub.getShadowValue()

    @shadow_value.setter
    def shadow_value(self, value: int) -> None:
        self._bgsub.setShadowValue(value)


class BackgroundSubtractorMOG2(BackgroundSubtractor):
    def __init__(
        self,
        history=DEFAULT_HISTORY,
        threshold=DEFAULT_MOG2_THRESHOLD,
        shadow=DEFAULT_DETECT_SHADOWS,
    ):
        super().__init__(BackgroundSubtractorMethod.MOG2, history, threshold, shadow)

        assert hasattr(self._bgsub, "getBackgroundRatio")
        assert hasattr(self._bgsub, "getComplexityReductionThreshold")
        assert hasattr(self._bgsub, "getDetectShadows")
        assert hasattr(self._bgsub, "getHistory")
        assert hasattr(self._bgsub, "getNMixtures")
        assert hasattr(self._bgsub, "getShadowThreshold")
        assert hasattr(self._bgsub, "getShadowValue")
        assert hasattr(self._bgsub, "getVarInit")
        assert hasattr(self._bgsub, "getVarMax")
        assert hasattr(self._bgsub, "getVarMin")
        assert hasattr(self._bgsub, "getVarThreshold")
        assert hasattr(self._bgsub, "getVarThresholdGen")

        assert hasattr(self._bgsub, "setBackgroundRatio")
        assert hasattr(self._bgsub, "setComplexityReductionThreshold")
        assert hasattr(self._bgsub, "setDetectShadows")
        assert hasattr(self._bgsub, "setHistory")
        assert hasattr(self._bgsub, "setNMixtures")
        assert hasattr(self._bgsub, "setShadowThreshold")
        assert hasattr(self._bgsub, "setShadowValue")
        assert hasattr(self._bgsub, "setVarInit")
        assert hasattr(self._bgsub, "setVarMax")
        assert hasattr(self._bgsub, "setVarMin")
        assert hasattr(self._bgsub, "setVarThreshold")
        assert hasattr(self._bgsub, "setVarThresholdGen")

    @property
    def background_ratio(self) -> float:
        """Returns the "background ratio" parameter of the algorithm"""
        return self._bgsub.getBackgroundRatio()

    @background_ratio.setter
    def background_ratio(self, value: float) -> None:
        self._bgsub.setBackgroundRatio(value)

    @property
    def complexity_reduction_threshold(self) -> float:
        """Returns the complexity reduction threshold"""
        return self._bgsub.getComplexityReductionThreshold()

    @complexity_reduction_threshold.setter
    def complexity_reduction_threshold(self, value: float) -> None:
        self._bgsub.setComplexityReductionThreshold(value)

    @property
    def detect_shadows(self) -> bool:
        """Returns the shadow detection flag"""
        return self._bgsub.getDetectShadows()

    @detect_shadows.setter
    def detect_shadows(self, value: bool) -> None:
        self._bgsub.setDetectShadows(value)

    @property
    def history(self) -> int:
        """Returns the number of last frames that affect the background model"""
        return self._bgsub.getHistory()

    @history.setter
    def history(self, value: int) -> None:
        self._bgsub.setHistory(value)

    @property
    def number_mixtures(self) -> int:
        """Returns the number of gaussian components in the background model"""
        return self._bgsub.getNMixtures()

    @number_mixtures.setter
    def number_mixtures(self, value: int) -> None:
        self._bgsub.setNMixtures(value)

    @property
    def shadow_threshold(self) -> float:
        """Returns the shadow threshold"""
        return self._bgsub.getShadowThreshold()

    @shadow_threshold.setter
    def shadow_threshold(self, value: float) -> None:
        self._bgsub.setShadowThreshold(value)

    @property
    def shadow_value(self) -> int:
        """Returns the shadow value"""
        return self._bgsub.getShadowValue()

    @shadow_value.setter
    def shadow_value(self, value: int) -> None:
        self._bgsub.setShadowValue(value)

    @property
    def var_init(self) -> float:
        """Returns the initial variance of each gaussian component"""
        return self._bgsub.getVarInit()

    @var_init.setter
    def var_init(self, value: float) -> None:
        self._bgsub.setVarInit(value)

    @property
    def var_max(self) -> float:
        return self._bgsub.getVarMax()

    @var_max.setter
    def var_max(self, value: float) -> None:
        self._bgsub.setVarMax(value)

    @property
    def var_min(self) -> float:
        return self._bgsub.getVarMin()

    @var_min.setter
    def var_min(self, value: float) -> None:
        self._bgsub.setVarMin(value)

    @property
    def var_threshold(self) -> float:
        """Returns the variance threshold for the pixel-model match"""
        return self._bgsub.getVarThreshold()

    @var_threshold.setter
    def var_threshold(self, value: float) -> None:
        self._bgsub.setVarThreshold(value)

    @property
    def var_threshold_gen(self) -> float:
        """
        Returns the variance threshold for the pixel-model match used
        for new mixture component generation.
        """
        return self._bgsub.getVarThresholdGen()

    @var_threshold_gen.setter
    def var_threshold_gen(self, value: float) -> None:
        self._bgsub.setVarThresholdGen(value)


class CvlBackgroundSubtractor:
    @staticmethod
    def cvl_create_background_subtractor_knn(
        history=DEFAULT_HISTORY,
        threshold=DEFAULT_KNN_THRESHOLD,
        shadow=DEFAULT_DETECT_SHADOWS,
    ):
        return BackgroundSubtractorKNN(history, threshold, shadow)

    @staticmethod
    def cvl_create_background_subtractor_mog2(
        history=DEFAULT_HISTORY,
        threshold=DEFAULT_MOG2_THRESHOLD,
        shadow=DEFAULT_DETECT_SHADOWS,
    ):
        return BackgroundSubtractorMOG2(history, threshold, shadow)


if __name__ == "__main__":
    print(f"Available background subtractor: {find_background_subtractor()}")

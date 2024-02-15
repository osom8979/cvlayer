# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final, NamedTuple, Optional

import cv2
from numpy import float32, ndarray, uint8
from numpy.typing import NDArray

from cvlayer.cv.term_criteria import TermCriteria, TermCriteriaType

DEFAULT_TERM_CRITERIA_TYPE: Final[TermCriteriaType] = TermCriteriaType.COUNT_EPS
DEFAULT_TERM_CRITERIA_MAX_COUNT: Final[int] = 10
DEFAULT_TERM_CRITERIA_EPSILON: Final[float] = 1.0

DEFAULT_TERM_CRITERIA: Final[TermCriteria] = TermCriteria(
    DEFAULT_TERM_CRITERIA_TYPE,
    DEFAULT_TERM_CRITERIA_MAX_COUNT,
    DEFAULT_TERM_CRITERIA_EPSILON,
)
DEFAULT_ATTEMPTS: Final[int] = 10


@unique
class KmeansFlags(Enum):
    RANDOM_CENTERS = cv2.KMEANS_RANDOM_CENTERS
    """
    Select random initial centers in each attempt.
    """

    PP_CENTERS = cv2.KMEANS_PP_CENTERS
    """
    Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007].
    """

    USE_INITIAL_LABELS = cv2.KMEANS_USE_INITIAL_LABELS
    """
    During the first (and possibly the only) attempt,
    use the user-supplied labels instead of computing them from the initial centers.
    For the second and further attempts, use the random or semi-random centers.
    Use one of `KMEANS_*_CENTERS` flag to specify the exact method.
    """


class KmeansResult(NamedTuple):
    retval: float
    labels: NDArray
    centers: NDArray


def kmeans(
    data: NDArray,
    k: int,
    best_labels: Optional[NDArray] = None,
    term_criteria=DEFAULT_TERM_CRITERIA,
    attempts=DEFAULT_ATTEMPTS,
    flags=KmeansFlags.PP_CENTERS,
) -> KmeansResult:
    criteria = (
        term_criteria.type.value,
        term_criteria.max_count,
        term_criteria.epsilon,
    )
    retval, labels, centers = cv2.kmeans(
        data=data,
        K=k,
        bestLabels=best_labels,  # type: ignore[arg-type]
        criteria=criteria,
        attempts=attempts,
        flags=flags.value,
        centers=None,
    )
    return KmeansResult(retval, labels, centers)


def color_quantization(
    image: NDArray,
    k: int,
    best_labels: Optional[NDArray] = None,
    criteria=DEFAULT_TERM_CRITERIA,
    attempts=DEFAULT_ATTEMPTS,
    flags=KmeansFlags.PP_CENTERS,
) -> NDArray:
    z = float32(image.reshape((-1, image.shape[-1])))
    assert isinstance(z, ndarray)
    result = kmeans(z, k, best_labels, criteria, attempts, flags)
    center = uint8(result.centers)
    flatten_labels = result.labels.flatten()
    return center[flatten_labels].reshape(image.shape)


class CvlKmeans:
    @staticmethod
    def cvl_kmeans(
        data: NDArray,
        k: int,
        best_labels: Optional[NDArray] = None,
        criteria=DEFAULT_TERM_CRITERIA,
        attempts=DEFAULT_ATTEMPTS,
        flags=KmeansFlags.PP_CENTERS,
    ):
        return kmeans(data, k, best_labels, criteria, attempts, flags)

    @staticmethod
    def cvl_color_quantization(
        image: NDArray,
        k: int,
        best_labels: Optional[NDArray] = None,
        criteria=DEFAULT_TERM_CRITERIA,
        attempts=DEFAULT_ATTEMPTS,
        flags=KmeansFlags.PP_CENTERS,
    ):
        return color_quantization(image, k, best_labels, criteria, attempts, flags)

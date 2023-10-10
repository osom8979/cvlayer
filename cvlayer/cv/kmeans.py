# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final, NamedTuple, Optional

import cv2
from numpy import float32, ndarray, uint8
from numpy.typing import NDArray

assert cv2.TERM_CRITERIA_COUNT == cv2.TERM_CRITERIA_MAX_ITER


@unique
class TermCriteriaType(Enum):
    COUNT = cv2.TERM_CRITERIA_COUNT
    """
    the maximum number of iterations or elements to compute
    """

    EPS = cv2.TERM_CRITERIA_EPS
    """
    the desired accuracy or change in parameters at which the iterative algorithm stops
    """

    COUNT_EPS = cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS
    """
    COUNT + EPS
    """


class TermCriteria(NamedTuple):
    type: TermCriteriaType
    """
    The type of termination criteria: COUNT, EPS or COUNT + EPS
    """

    max_count: int
    """
    The maximum number of iterations/elements
    """

    epsilon: float
    """
    The desired accuracy
    """


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


DEFAULT_CRITERIA_TYPE: Final[TermCriteriaType] = TermCriteriaType.COUNT_EPS
DEFAULT_CRITERIA_MAX_COUNT: Final[int] = 10
DEFAULT_CRITERIA_EPSILON: Final[float] = 1.0
DEFAULT_CRITERIA: Final[TermCriteria] = TermCriteria(
    DEFAULT_CRITERIA_TYPE,
    DEFAULT_CRITERIA_MAX_COUNT,
    DEFAULT_CRITERIA_EPSILON,
)
DEFAULT_ATTEMPTS: Final[int] = 10


def kmeans(
    data: NDArray,
    k: int,
    best_labels: Optional[NDArray] = None,
    criteria=DEFAULT_CRITERIA,
    attempts=DEFAULT_ATTEMPTS,
    flags=KmeansFlags.PP_CENTERS,
) -> KmeansResult:
    retval, labels, centers = cv2.kmeans(
        data,
        k,
        best_labels,
        (criteria.type.value, criteria.max_count, criteria.epsilon),
        attempts,
        flags.value,
    )
    return KmeansResult(retval, labels, centers)


def color_quantization(
    image: NDArray,
    k: int,
    best_labels: Optional[NDArray] = None,
    criteria=DEFAULT_CRITERIA,
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
        criteria=DEFAULT_CRITERIA,
        attempts=DEFAULT_ATTEMPTS,
        flags=KmeansFlags.PP_CENTERS,
    ):
        return kmeans(data, k, best_labels, criteria, attempts, flags)

    @staticmethod
    def cvl_color_quantization(
        image: NDArray,
        k: int,
        best_labels: Optional[NDArray] = None,
        criteria=DEFAULT_CRITERIA,
        attempts=DEFAULT_ATTEMPTS,
        flags=KmeansFlags.PP_CENTERS,
    ):
        return color_quantization(image, k, best_labels, criteria, attempts, flags)

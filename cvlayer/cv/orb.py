# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final, Sequence, Tuple

import cv2
from numpy.typing import NDArray


@unique
class OrbScoreType(Enum):
    HARRIS_SCORE = cv2.ORB_HARRIS_SCORE
    FAST_SCORE = cv2.ORB_FAST_SCORE


ORB_CREATE_FUNC: Final[str] = "ORB_create"
DEFAULT_N_FEATURES: Final[int] = 500
DEFAULT_SCALE_FACTOR: Final[float] = 1.2
DEFAULT_N_LEVELS: Final[int] = 8
DEFAULT_EDGE_THRESHOLD: Final[int] = 31
DEFAULT_FIRST_LEVEL: Final[int] = 0
DEFAULT_WTA_K: Final[int] = 2
DEFAULT_SCORE_TYPE: Final[OrbScoreType] = OrbScoreType.HARRIS_SCORE
DEFAULT_PATCH_SIZE: Final[int] = 31
DEFAULT_FAST_THRESHOLD: Final[int] = 20


def create_orb(*args, **kwargs):
    """
    https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html
    """

    assert hasattr(cv2, ORB_CREATE_FUNC)
    func = getattr(cv2, ORB_CREATE_FUNC)
    return func(*args, **kwargs)


class Orb:
    def __init__(
        self,
        n_features=DEFAULT_N_FEATURES,
        scale_factor=DEFAULT_SCALE_FACTOR,
        n_levels=DEFAULT_N_LEVELS,
        edge_threshold=DEFAULT_EDGE_THRESHOLD,
        first_level=DEFAULT_FIRST_LEVEL,
        wta_k=DEFAULT_WTA_K,
        score_type=DEFAULT_SCORE_TYPE,
        patch_size=DEFAULT_PATCH_SIZE,
        fast_threshold=DEFAULT_FAST_THRESHOLD,
    ):
        self._orb = create_orb(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            firstLevel=first_level,
            WTA_K=wta_k,
            scoreType=score_type.value,
            patchSize=patch_size,
            fastThreshold=fast_threshold,
        )

        assert hasattr(self._orb, "detectAndCompute")

    def detect_and_compute(
        self,
        image: NDArray,
    ) -> Tuple[Sequence[cv2.KeyPoint], NDArray]:
        return self._orb.detectAndCompute(image, None)


class CvlOrb:
    @staticmethod
    def cvl_create_orb(
        n_features=DEFAULT_N_FEATURES,
        scale_factor=DEFAULT_SCALE_FACTOR,
        n_levels=DEFAULT_N_LEVELS,
        edge_threshold=DEFAULT_EDGE_THRESHOLD,
        first_level=DEFAULT_FIRST_LEVEL,
        wta_k=DEFAULT_WTA_K,
        score_type=DEFAULT_SCORE_TYPE,
        patch_size=DEFAULT_PATCH_SIZE,
        fast_threshold=DEFAULT_FAST_THRESHOLD,
    ):
        return Orb(
            n_features,
            scale_factor,
            n_levels,
            edge_threshold,
            first_level,
            wta_k,
            score_type,
            patch_size,
            fast_threshold,
        )

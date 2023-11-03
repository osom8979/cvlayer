# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

import cv2

CHAIN_APPROX_NONE: Final[int] = cv2.CHAIN_APPROX_NONE
CHAIN_APPROX_SIMPLE: Final[int] = cv2.CHAIN_APPROX_SIMPLE
CHAIN_APPROX_TC89_KCOS: Final[int] = cv2.CHAIN_APPROX_TC89_KCOS
CHAIN_APPROX_TC89_L1: Final[int] = cv2.CHAIN_APPROX_TC89_L1


@unique
class ChainApproximation(Enum):
    NONE = CHAIN_APPROX_NONE
    SIMPLE = CHAIN_APPROX_SIMPLE
    TC89_KCOS = CHAIN_APPROX_TC89_KCOS
    TC89_L1 = CHAIN_APPROX_TC89_L1


ChainApproximationLike = Union[ChainApproximation, str, int]

DEFAULT_CHAIN_APPROX: Final[ChainApproximationLike] = CHAIN_APPROX_SIMPLE
CHAIN_APPROX_TYPE_MAP: Final[Dict[str, int]] = {
    # ChainApproximation enum names
    "NONE": CHAIN_APPROX_NONE,
    "SIMPLE": CHAIN_APPROX_SIMPLE,
    "TC89_KCOS": CHAIN_APPROX_TC89_KCOS,
    "TC89_L1": CHAIN_APPROX_TC89_L1,
    # cv2 symbol full names
    "CHAIN_APPROX_NONE": CHAIN_APPROX_NONE,
    "CHAIN_APPROX_SIMPLE": CHAIN_APPROX_SIMPLE,
    "CHAIN_APPROX_TC89_KCOS": CHAIN_APPROX_TC89_KCOS,
    "CHAIN_APPROX_TC89_L1": CHAIN_APPROX_TC89_L1,
}


def normalize_chain_approx(chain_approx: Optional[ChainApproximationLike]) -> int:
    if chain_approx is None:
        assert isinstance(DEFAULT_CHAIN_APPROX, int)
        return DEFAULT_CHAIN_APPROX

    if isinstance(chain_approx, ChainApproximation):
        return chain_approx.value
    elif isinstance(chain_approx, str):
        return CHAIN_APPROX_TYPE_MAP[chain_approx.upper()]
    elif isinstance(chain_approx, int):
        return chain_approx
    else:
        raise TypeError(f"Unsupported chain-approx type: {type(chain_approx).__name__}")

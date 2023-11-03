# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

import cv2

RETR_CCOMP: Final[int] = cv2.RETR_CCOMP
RETR_EXTERNAL: Final[int] = cv2.RETR_EXTERNAL
RETR_LIST: Final[int] = cv2.RETR_LIST
RETR_TREE: Final[int] = cv2.RETR_TREE
RETR_FLOODFILL: Final[int] = cv2.RETR_FLOODFILL


@unique
class Retrieval(Enum):
    CCOMP = RETR_CCOMP
    EXTERNAL = RETR_EXTERNAL
    LIST = RETR_LIST
    TREE = RETR_TREE
    FLOODFILL = RETR_FLOODFILL


RetrievalLike = Union[Retrieval, str, int]

DEFAULT_RETRIEVAL: Final[RetrievalLike] = RETR_TREE
RETRIEVAL_TYPE_MAP: Final[Dict[str, int]] = {
    # Retrieval enum names
    "CCOMP": RETR_CCOMP,
    "EXTERNAL": RETR_EXTERNAL,
    "LIST": RETR_LIST,
    "TREE": RETR_TREE,
    "FLOODFILL": RETR_FLOODFILL,
    # cv2 symbol full names
    "RETR_CCOMP": RETR_CCOMP,
    "RETR_EXTERNAL": RETR_EXTERNAL,
    "RETR_LIST": RETR_LIST,
    "RETR_TREE": RETR_TREE,
    "RETR_FLOODFILL": RETR_FLOODFILL,
}


def normalize_retrieval(retrieval: Optional[RetrievalLike]) -> int:
    if retrieval is None:
        assert isinstance(DEFAULT_RETRIEVAL, int)
        return DEFAULT_RETRIEVAL

    if isinstance(retrieval, Retrieval):
        return retrieval.value
    elif isinstance(retrieval, str):
        return RETRIEVAL_TYPE_MAP[retrieval.upper()]
    elif isinstance(retrieval, int):
        return retrieval
    else:
        raise TypeError(f"Unsupported retrieval type: {type(retrieval).__name__}")

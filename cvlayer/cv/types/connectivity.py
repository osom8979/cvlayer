# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

CONNECTIVITY4: Final[int] = 4
CONNECTIVITY8: Final[int] = 8


@unique
class Connectivity(Enum):
    C4 = CONNECTIVITY4
    C8 = CONNECTIVITY8


ConnectivityLike = Union[Connectivity, str, int]

DEFAULT_CONNECTIVITY: Final[ConnectivityLike] = CONNECTIVITY4
CONNECTIVITY_MAP: Final[Dict[str, int]] = {
    # str to int names
    "4": CONNECTIVITY4,
    "8": CONNECTIVITY8,
    # Connectivity enum names
    "C4": CONNECTIVITY4,
    "C8": CONNECTIVITY8,
    # global constant names
    "CONNECTIVITY4": CONNECTIVITY4,
    "CONNECTIVITY8": CONNECTIVITY8,
}


def normalize_connectivity(connectivity: Optional[ConnectivityLike]) -> int:
    if connectivity is None:
        assert isinstance(DEFAULT_CONNECTIVITY, int)
        return DEFAULT_CONNECTIVITY  # type: ignore[return-value]  # mypy bug ?

    if isinstance(connectivity, Connectivity):
        return connectivity.value
    elif isinstance(connectivity, str):
        return CONNECTIVITY_MAP[connectivity.upper()]
    elif isinstance(connectivity, int):
        return connectivity
    else:
        raise TypeError(f"Unsupported connectivity type: {type(connectivity).__name__}")

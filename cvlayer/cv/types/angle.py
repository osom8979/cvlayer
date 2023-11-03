# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, Optional, Union

ANGLE_IN_DEGREES: Final[bool] = True
MEASURED_IN_RADIANS: Final[bool] = False


@unique
class AngleType(Enum):
    DEGREES = ANGLE_IN_DEGREES
    RADIANS = MEASURED_IN_RADIANS


AngleTypeLike = Union[AngleType, str, int, bool]

DEFAULT_ANGLE_TYPE: Final[AngleTypeLike] = MEASURED_IN_RADIANS
ANGLE_TYPE_MAP: Final[Dict[str, bool]] = {
    "DEGREES": ANGLE_IN_DEGREES,
    "RADIANS": MEASURED_IN_RADIANS,
}


def normalize_angle_type(angle_type: Optional[AngleTypeLike]) -> bool:
    if angle_type is None:
        assert isinstance(DEFAULT_ANGLE_TYPE, bool)
        return DEFAULT_ANGLE_TYPE  # type: ignore[return-value]  # mypy bug ?

    if isinstance(angle_type, AngleType):
        return angle_type.value
    elif isinstance(angle_type, str):
        return ANGLE_TYPE_MAP[angle_type.upper()]
    elif isinstance(angle_type, int):
        return angle_type != 0
    elif isinstance(angle_type, bool):
        return angle_type
    else:
        raise TypeError(f"Unsupported angle type: {type(angle_type).__name__}")

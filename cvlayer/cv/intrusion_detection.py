# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, Final, List, Optional, Union

from numpy import ndarray
from numpy.typing import NDArray
from shapely.geometry import Polygon

from cvlayer.cv.contours_intersection import intersection_polygon_and_polygon
from cvlayer.cv.cvt_shapely import (
    cvt_contour2polygon,
    cvt_polygon2contour,
    cvt_roi2contour,
    cvt_roi2polygon,
)
from cvlayer.types import RectT

AreaType = Union[NDArray, RectT]

ANONYMOUS_INSTANCE_UID: Final[int] = -1

COUNTER_CONFIG_DIRECT_THRESHOLD: Final[int] = 1
COUNTER_CONFIG_DIRECT_MAXVALUE: Final[int] = 1
COUNTER_CONFIG_DIRECT_MINVALUE: Final[int] = 0
COUNTER_CONFIG_DIRECT_UPGRADE_PERIOD: Final[int] = 1
COUNTER_CONFIG_DIRECT_DOWNGRADE_PERIOD: Final[int] = 1
COUNTER_CONFIG_DIRECT_UPGRADE_INIT: Final[int] = 1
COUNTER_CONFIG_DIRECT_DOWNGRADE_INIT: Final[int] = 0


class _GradeDone(Exception):
    pass


class _MaximumUpgrade(_GradeDone):
    pass


class _MinimumDowngrade(_GradeDone):
    pass


def cvt_area2polygon(area: AreaType) -> Polygon:
    if isinstance(area, ndarray):
        return cvt_contour2polygon(area)
    else:
        assert isinstance(area, (tuple, list))
        return cvt_roi2polygon(area)


class Realm:
    def __init__(self, area: AreaType):
        self._contour = area if isinstance(area, ndarray) else cvt_roi2contour(area)
        self._polygon = cvt_area2polygon(area)

    @property
    def contour(self):
        return self._contour

    @property
    def polygon(self):
        return self._polygon

    def intersection(self, polygon: Polygon) -> List[Polygon]:
        return intersection_polygon_and_polygon(self._polygon, polygon)


@dataclass
class CounterConfig:
    threshold: int = 1
    maxvalue: int = 1
    minvalue: int = 0
    upgrade_period: int = 1
    downgrade_period: int = 1
    upgrade_init: int = 1
    downgrade_init: int = 0

    @classmethod
    def from_unit(cls, step=2, period=1, thresh_weight=2, maxvalue_weight=3):
        assert step >= 1
        assert period >= 1
        assert thresh_weight >= 1
        assert maxvalue_weight >= 1
        assert maxvalue_weight > thresh_weight

        unit = period * step
        threshold = unit * thresh_weight
        maxvalue = unit * maxvalue_weight
        minvalue = 0
        upgrade_period = period
        downgrade_period = period
        upgrade_init = minvalue + unit
        downgrade_init = threshold

        return cls(
            threshold=threshold,
            maxvalue=maxvalue,
            minvalue=minvalue,
            upgrade_period=upgrade_period,
            downgrade_period=downgrade_period,
            upgrade_init=upgrade_init,
            downgrade_init=downgrade_init,
        )


class ObjectState:
    _first_intrusions: List[Polygon]
    _second_intrusions: List[Polygon]

    def __init__(self, config: CounterConfig, max_level: int):
        self._config = config
        self._max_level = max_level

        self._first_intrusions = list()
        self._second_intrusions = list()

        self._level = 0
        self._counter = config.minvalue

    @property
    def level(self) -> int:
        return self._level

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def is_max_level(self) -> bool:
        assert 0 <= self._level <= self._max_level
        return self._level == self._max_level

    @property
    def is_min_level(self) -> bool:
        assert 0 <= self._level <= self._max_level
        return self._level == 0

    @property
    def is_current_level_inbound(self) -> bool:
        return self._counter >= self._config.threshold

    @property
    def is_current_level_outbound(self) -> bool:
        return not self.is_current_level_inbound

    @property
    def is_max_inbound(self) -> bool:
        return self.is_max_level and self.is_current_level_inbound

    @property
    def is_min_outbound(self) -> bool:
        return self.is_min_level and self.is_current_level_outbound

    @property
    def as_first_intrusions(self) -> List[NDArray]:
        return list(map(lambda p: cvt_polygon2contour(p), self._first_intrusions))

    @property
    def as_second_intrusions(self) -> List[NDArray]:
        return list(map(lambda p: cvt_polygon2contour(p), self._second_intrusions))

    def upgrade(self, area: Polygon, realms: List[Realm]) -> None:
        self._counter += self._config.upgrade_period

        if self._counter > self._config.maxvalue:
            self._counter = self._config.maxvalue

        if self._counter < self._config.threshold:
            raise _GradeDone

        assert self._counter >= self._config.threshold
        up_level = self._level + 1

        if up_level >= len(realms):
            raise _MaximumUpgrade

        assert 0 <= up_level < len(realms)
        self._second_intrusions = realms[up_level].intersection(area)

        if self._second_intrusions:
            self._level = up_level
            self._counter = self._config.upgrade_init

    def downgrade(self, area: Polygon, realms: List[Realm]) -> None:
        self._counter -= self._config.downgrade_period

        if self._counter >= self._config.minvalue:
            raise _GradeDone

        assert self._counter < self._config.minvalue
        if self._level == 0:
            self._counter = self._config.minvalue
            raise _MinimumDowngrade

        down_level = self._level - 1
        assert 0 <= down_level < len(realms)

        self._level = down_level
        self._second_intrusions = realms[down_level].intersection(area)
        if self._second_intrusions:
            self._counter = self._config.threshold
        else:
            self._counter = self._config.downgrade_init

    def do_grade(self, area: Polygon, realms: List[Realm]) -> None:
        self._first_intrusions = realms[self._level].intersection(area)
        try:
            if self._first_intrusions:
                return self.upgrade(area, realms)
            else:
                return self.downgrade(area, realms)
        except _GradeDone:
            pass


class HierarchicalIntrusionDetection:
    _realms: List[Realm]
    _states: Dict[int, ObjectState]

    def __init__(
        self,
        *areas: AreaType,
        config: Optional[CounterConfig] = None,
    ):
        self._config = config if config else CounterConfig()
        self._realms = [Realm(area) for area in areas]
        self._states = dict()

        if not self._realms:
            raise IndexError("Realm does not exist")

    @property
    def config(self):
        return self._config

    @property
    def realms(self):
        return self._realms

    @property
    def states(self):
        return self._states

    @property
    def max_level(self) -> int:
        return len(self._realms) - 1

    def get_state(self, uid: int) -> ObjectState:
        if uid not in self._states:
            self._states[uid] = ObjectState(self.config, self.max_level)
        return self._states[uid]

    def run(self, area: AreaType, uid=ANONYMOUS_INSTANCE_UID) -> ObjectState:
        if not self._realms:
            raise IndexError("Realm does not exist")

        polygon = cvt_area2polygon(area)
        state = self.get_state(uid)
        state.do_grade(polygon, self._realms)
        return state


class CvlIntrusionDetection:
    @staticmethod
    def cvl_cvt_area2polygon(area: AreaType):
        return cvt_area2polygon(area)

    @staticmethod
    def cvl_create_counter_config(
        threshold=1,
        maxvalue=1,
        minvalue=0,
        upgrade_period=1,
        downgrade_period=1,
        upgrade_init=1,
        downgrade_init=0,
    ):
        return CounterConfig(
            threshold,
            maxvalue,
            minvalue,
            upgrade_period,
            downgrade_period,
            upgrade_init,
            downgrade_init,
        )

    @staticmethod
    def cvl_create_counter_config_from_unit(
        step=2,
        period=1,
        thresh_weight=2,
        maxvalue_weight=3,
    ):
        return CounterConfig.from_unit(
            step,
            period,
            thresh_weight,
            maxvalue_weight,
        )

    @staticmethod
    def cvl_create_hierarchical_intrusion_detection(
        *areas: AreaType,
        config: Optional[CounterConfig] = None,
    ):
        return HierarchicalIntrusionDetection(*areas, config=config)

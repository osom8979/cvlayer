# -*- coding: utf-8 -*-

from copy import deepcopy
from enum import Enum
from math import ceil, floor
from typing import Any, Callable, Iterable, Optional

LimitedCallable = Callable[[], Any]
ModifyCallable = Callable[[Any], Any]
PrintableCallable = Callable[[Any], str]
GetterCallable = Callable[[Any], Any]
SetterCallable = Callable[[Any], Any]


class LayerParameter:
    def __init__(
        self,
        value: Any,
        min_value: Optional[LimitedCallable] = None,
        max_value: Optional[LimitedCallable] = None,
        decrease: Optional[ModifyCallable] = None,
        increase: Optional[ModifyCallable] = None,
        getter: Optional[GetterCallable] = None,
        setter: Optional[SetterCallable] = None,
        printable: Optional[PrintableCallable] = None,
    ):
        self._value = value
        self._min_value = min_value
        self._max_value = max_value
        self._decrease = decrease
        self._increase = increase
        self._getter = getter
        self._setter = setter
        self._printable = printable

    def normalize_by_candidate_value(self, value: Any) -> Any:
        if not isinstance(value, type(self._value)):
            raise TypeError(f"The value must be of type {type(self._value).__name__}")

        if value is None:
            return None

        minval = self._min_value() if self._min_value else None
        maxval = self._max_value() if self._max_value else None

        if minval is not None and value < minval:
            return minval
        if maxval is not None and value > maxval:
            return maxval
        else:
            return value

    @property
    def value(self) -> Any:
        if self._getter:
            return self._getter(self._value)
        else:
            return self._value

    @value.setter
    def value(self, value: Any) -> None:
        normalized = self.normalize_by_candidate_value(value)
        if self._setter:
            self._value = self._setter(normalized)
        else:
            self._value = normalized

    def decrease(self) -> None:
        if not self._decrease:
            return
        candidate = self._decrease(self._value)
        self._value = self.normalize_by_candidate_value(candidate)

    def increase(self) -> None:
        if not self._increase:
            return
        candidate = self._increase(self._value)
        self._value = self.normalize_by_candidate_value(candidate)

    def printable(self) -> str:
        if self._printable:
            return self._printable(self._value)
        else:
            return str(self._value)

    def __str__(self):
        return str(self._value)

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)

    def __floor__(self):
        return floor(self._value)

    def __ceil__(self):
        return ceil(self._value)

    def __repr__(self):
        return self.printable()

    @classmethod
    def make_printonly(cls, printable: Callable[[], str]):
        return cls(value=None, printable=lambda x: printable())

    @classmethod
    def make_readonly(cls, value: Any, printable: Optional[PrintableCallable] = None):
        return cls(value=value, printable=printable)

    @classmethod
    def make_bool(cls, value: bool, printable: Optional[PrintableCallable] = None):
        return cls(
            value=value,
            decrease=lambda x: not x,
            increase=lambda x: not x,
            printable=printable,
        )

    @classmethod
    def make_int(
        cls,
        value: int,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        printable: Optional[PrintableCallable] = None,
        step=1,
    ):
        return cls(
            value=value,
            min_value=lambda: min_value,
            max_value=lambda: max_value,
            decrease=lambda x: x - step,
            increase=lambda x: x + step,
            printable=printable,
        )

    @classmethod
    def make_unsigned(
        cls,
        value: int,
        max_value: Optional[int] = None,
        printable: Optional[PrintableCallable] = None,
        step=1,
    ):
        return cls.make_int(
            value=value,
            min_value=0,
            max_value=max_value,
            printable=printable,
            step=step,
        )

    @classmethod
    def make_float(
        cls,
        value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        printable: Optional[PrintableCallable] = None,
        step=0.01,
    ):
        return cls(
            value=value,
            min_value=lambda: min_value,
            max_value=lambda: max_value,
            decrease=lambda x: x - step,
            increase=lambda x: x + step,
            printable=printable,
        )

    @classmethod
    def make_enum(cls, value: Enum):
        assert isinstance(value, Enum)
        enum_cls = type(value)
        assert issubclass(enum_cls, Enum)
        available_enums = [m for m in enum_cls]
        return cls(
            value=available_enums.index(value),
            min_value=lambda: 0,
            max_value=lambda: len(available_enums) - 1,
            decrease=lambda index: index - 1,
            increase=lambda index: index + 1,
            getter=lambda index: available_enums[index],
            setter=lambda element: available_enums.index(element),
            printable=lambda index: f"{index} ({available_enums[index].name})",
        )

    @classmethod
    def make_list(cls, items: Iterable[Any], value: Optional[Any] = None):
        available_items = list(deepcopy(items))
        return cls(
            value=available_items.index(value),
            min_value=lambda: 0,
            max_value=lambda: len(available_items) - 1,
            decrease=lambda index: index - 1,
            increase=lambda index: index + 1,
            getter=lambda index: available_items[index],
            setter=lambda element: available_items.index(element),
            printable=lambda index: f"{index} ({available_items[index]})",
        )

# -*- coding: utf-8 -*-

from copy import deepcopy
from enum import Enum
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
        return self._getter(self._value) if self._getter else self._value

    @value.setter
    def value(self, value: Any) -> None:
        normalized_val = self.normalize_by_candidate_value(value)
        self._value = self._setter(normalized_val) if self._setter else normalized_val

    def decrease(self) -> None:
        if not self._decrease:
            return
        self._value = self.normalize_by_candidate_value(self._decrease(self._value))

    def increase(self) -> None:
        if not self._increase:
            return
        self._value = self.normalize_by_candidate_value(self._increase(self._value))

    def printable(self) -> str:
        return self._printable(self._value) if self._printable else str(self._value)


def printonly_param(printable: Callable[[], str]) -> LayerParameter:
    return LayerParameter(value=None, printable=lambda x: printable())


def readonly_param(
    value: Any,
    printable: Optional[PrintableCallable] = None,
) -> LayerParameter:
    return LayerParameter(value=value, printable=printable)


def bool_param(
    value: bool,
    printable: Optional[PrintableCallable] = None,
) -> LayerParameter:
    return LayerParameter(
        value=value,
        decrease=lambda x: not x,
        increase=lambda x: not x,
        printable=printable,
    )


def int_param(
    value: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    printable: Optional[PrintableCallable] = None,
    step=1,
) -> LayerParameter:
    return LayerParameter(
        value=value,
        min_value=lambda: min_value,
        max_value=lambda: max_value,
        decrease=lambda x: x - step,
        increase=lambda x: x + step,
        printable=printable,
    )


def uint_param(
    value: int,
    max_value: Optional[int] = None,
    step=1,
) -> LayerParameter:
    return int_param(
        value=value,
        min_value=0,
        max_value=max_value,
        printable=None,
        step=step,
    )


def float_param(
    value: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    printable: Optional[PrintableCallable] = None,
    step=0.1,
) -> LayerParameter:
    return LayerParameter(
        value=value,
        min_value=lambda: min_value,
        max_value=lambda: max_value,
        decrease=lambda x: x - step,
        increase=lambda x: x + step,
        printable=printable,
    )


def enum_param(value: Enum) -> LayerParameter:
    assert isinstance(value, Enum)
    enum_cls = type(value)
    assert issubclass(enum_cls, Enum)
    available_enums = [m for m in enum_cls]
    return LayerParameter(
        value=available_enums.index(value),
        min_value=lambda: 0,
        max_value=lambda: len(available_enums) - 1,
        decrease=lambda index: index - 1,
        increase=lambda index: index + 1,
        getter=lambda index: available_enums[index],
        setter=lambda element: available_enums.index(element),
        printable=lambda index: f"{index} ({available_enums[index].name})",
    )


def list_param(items: Iterable[Any], value: Optional[Any] = None) -> LayerParameter:
    available_items = list(deepcopy(items))
    return LayerParameter(
        value=available_items.index(value),
        min_value=lambda: 0,
        max_value=lambda: len(available_items) - 1,
        decrease=lambda index: index - 1,
        increase=lambda index: index + 1,
        getter=lambda index: available_items[index],
        setter=lambda element: available_items.index(element),
        printable=lambda index: f"{index} ({available_items[index]})",
    )

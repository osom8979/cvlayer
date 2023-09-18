# -*- coding: utf-8 -*-

from copy import deepcopy
from enum import Enum
from math import ceil, floor
from typing import Any, Callable, Iterable, Optional, Union

from cvlayer.cv.mouse import EventFlags, MouseEvent
from cvlayer.typing import PointInt, RectInt

LimitedCallable = Callable[[], Any]
ModifyCallable = Callable[[Any], Any]
PrintableCallable = Callable[[Any], str]
GetterCallable = Callable[[Any], Any]
SetterCallable = Callable[[Any], Any]
OnKeydownCallable = Callable[[int], Optional[bool]]
OnMouseCallable = Callable[[MouseEvent, int, int, EventFlags], Optional[bool]]


class AlreadyFrozenError(BaseException):
    def __init__(self, *args):
        super().__init__(*args)


class NotReadyFrozenError(BaseException):
    def __init__(self, *args):
        super().__init__(*args)


class LayerParameter:
    def __init__(
        self,
        value=None,
        min_value: Optional[LimitedCallable] = None,
        max_value: Optional[LimitedCallable] = None,
        decrease: Optional[ModifyCallable] = None,
        increase: Optional[ModifyCallable] = None,
        getter: Optional[GetterCallable] = None,
        setter: Optional[SetterCallable] = None,
        printable: Optional[PrintableCallable] = None,
        keydown: Optional[OnKeydownCallable] = None,
        mouse: Optional[OnMouseCallable] = None,
        nullable=True,
        frozen=False,
        **kwargs,
    ):
        self._value = value
        self._min_value = min_value
        self._max_value = max_value
        self._decrease = decrease
        self._increase = increase
        self._getter = getter
        self._setter = setter
        self._printable = printable
        self._keydown = keydown
        self._mouse = mouse
        self._nullable = nullable
        self._frozen = frozen
        self._kwargs = kwargs

    def freeze(self) -> None:
        self._frozen = True

    def melt(self) -> None:
        self._frozen = False

    @property
    def initialized(self) -> bool:
        return self._frozen

    def _clear_all_properties(self) -> None:
        self._value = None
        self._min_value = None
        self._max_value = None
        self._decrease = None
        self._increase = None
        self._getter = None
        self._setter = None
        self._printable = None
        self._keydown = None
        self._mouse = None
        self._nullable = True
        self._frozen = False
        self._kwargs = dict()

    def _validate_initialize(self) -> None:
        if self._frozen:
            raise AlreadyFrozenError("Already frozen state")

    def _set_value(self, value: Any):
        self._validate_initialize()
        self._value = value

    def _set_min(self, callback: LimitedCallable):
        self._validate_initialize()
        self._min_value = callback

    def _set_max(self, callback: LimitedCallable):
        self._validate_initialize()
        self._max_value = callback

    def _set_decrease(self, callback: ModifyCallable):
        self._validate_initialize()
        self._decrease = callback

    def _set_increase(self, callback: ModifyCallable):
        self._validate_initialize()
        self._increase = callback

    def _set_getter(self, callback: GetterCallable):
        self._validate_initialize()
        self._getter = callback

    def _set_setter(self, callback: SetterCallable):
        self._validate_initialize()
        self._setter = callback

    def _set_keydown(self, callback: OnKeydownCallable):
        self._validate_initialize()
        self._keydown = callback

    def _set_mouse(self, callback: OnMouseCallable):
        self._validate_initialize()
        self._mouse = callback

    def _set_printable(self, callback: PrintableCallable):
        self._validate_initialize()
        self._printable = callback

    def _set_nullable(self, callback: PrintableCallable):
        self._validate_initialize()
        self._nullable = callback

    initial = property(None, _set_value)
    min = property(None, _set_min)
    max = property(None, _set_max)
    decrease = property(None, _set_decrease)
    increase = property(None, _set_increase)
    getter = property(None, _set_getter)
    setter = property(None, _set_setter)
    keydown = property(None, _set_keydown)
    mouse = property(None, _set_mouse)
    printable = property(None, _set_printable)
    nullable = property(None, _set_nullable)

    def normalize_by_candidate_value(self, value: Any) -> Any:
        if value is None:
            if self._nullable:
                return None
            else:
                raise ValueError("A 'None' value cannot be assigned.")

        assert value is not None
        if not isinstance(value, type(self._value)):
            raise TypeError(f"The value must be of type {type(self._value).__name__}")

        minval = self._min_value() if self._min_value else None
        maxval = self._max_value() if self._max_value else None

        if minval is not None and value < minval:
            return minval
        if maxval is not None and value > maxval:
            return maxval
        else:
            return value

    def validate(self) -> None:
        if not self._frozen:
            raise NotReadyFrozenError("Not ready frozen state")

    @property
    def value(self) -> Any:
        self.validate()
        if self._getter:
            return self._getter(self._value)
        else:
            return self._value

    @value.setter
    def value(self, val: Any) -> None:
        self.validate()
        normalized = self.normalize_by_candidate_value(val)
        if self._setter:
            self._value = self._setter(normalized)
        else:
            self._value = normalized

    def do_decrease(self) -> None:
        self.validate()
        if not self._decrease:
            return
        candidate = self._decrease(self._value)
        self._value = self.normalize_by_candidate_value(candidate)

    def do_increase(self) -> None:
        self.validate()
        if not self._increase:
            return
        candidate = self._increase(self._value)
        self._value = self.normalize_by_candidate_value(candidate)

    def as_printable_text(self) -> str:
        self.validate()
        if self._printable:
            return self._printable(self._value)
        else:
            return str(self._value)

    @property
    def has_keydown(self) -> bool:
        return self._keydown is not None

    @property
    def has_mouse(self) -> bool:
        return self._mouse is not None

    def call_keydown(self, keycode: int) -> Optional[bool]:
        if self._keydown:
            return self._keydown(keycode)
        else:
            return False

    def call_mouse(
        self,
        event: MouseEvent,
        x: int,
        y: int,
        flags: EventFlags,
    ) -> Optional[bool]:
        if self._mouse:
            return self._mouse(event, x, y, flags)
        else:
            return False

    def __str__(self):
        return str(self.value)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __floor__(self):
        return floor(self.value)

    def __ceil__(self):
        return ceil(self.value)

    def __repr__(self):
        return self.as_printable_text()

    def build_printonly(self, printable: Callable[[], str]):
        if self._frozen:
            return self
        self._clear_all_properties()
        self._printable = lambda x: printable()
        self._frozen = True
        return self

    def build_readonly(self, value: Any, printable: Optional[PrintableCallable] = None):
        if self._frozen:
            return self
        self._clear_all_properties()
        self._value = value
        self._printable = printable
        self._frozen = True
        return self

    def build_boolean(self, value: bool, printable: Optional[PrintableCallable] = None):
        if self._frozen:
            return self
        self._clear_all_properties()
        self._value = value
        self._decrease = lambda x: not x
        self._increase = lambda x: not x
        self._printable = printable
        self._frozen = True
        return self

    def build_integer(
        self,
        value: int,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        printable: Optional[PrintableCallable] = None,
        step=1,
    ):
        if self._frozen:
            return self
        self._clear_all_properties()
        self._value = value
        self._min_value = lambda: min_value
        self._max_value = lambda: max_value
        self._decrease = lambda x: x - step
        self._increase = lambda x: x + step
        self._printable = printable
        self._frozen = True
        return self

    def build_unsigned(
        self,
        value: int,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        printable: Optional[PrintableCallable] = None,
        step=1,
    ):
        if self._frozen:
            return self
        if min_value is not None and min_value < 0:
            raise ValueError("The unsigned type must be 0 or greater")
        return self.build_integer(
            value=value,
            min_value=min_value if min_value is not None else 0,
            max_value=max_value,
            printable=printable,
            step=step,
        )

    def build_floating(
        self,
        value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        printable: Optional[PrintableCallable] = None,
        step=0.01,
    ):
        if self._frozen:
            return self
        self._clear_all_properties()
        self._value = value
        self._min_value = lambda: min_value
        self._max_value = lambda: max_value
        self._decrease = lambda x: x - step
        self._increase = lambda x: x + step
        self._printable = printable
        self._frozen = True
        return self

    def build_enumeration(
        self,
        value: Enum,
        excludes: Optional[Union[Enum, Iterable[Enum]]] = None,
    ):
        if self._frozen:
            return self

        self._clear_all_properties()

        assert isinstance(value, Enum)
        enum_type = type(value)
        assert issubclass(enum_type, Enum)

        available_items = [m for m in enum_type]
        if excludes:
            if isinstance(excludes, enum_type):
                try:
                    available_items.remove(excludes)
                except ValueError:
                    pass
            elif isinstance(excludes, (tuple, list)):
                for ex in excludes:
                    try:
                        available_items.remove(ex)
                    except ValueError:
                        pass
            else:
                raise TypeError(f"Invalid exclude type: {type(excludes).__name__}")

        if not available_items:
            raise ValueError("There are no items available")

        self._value = available_items.index(value)
        self._min_value = lambda: 0
        self._max_value = lambda: len(available_items) - 1
        self._decrease = lambda index: index - 1
        self._increase = lambda index: index + 1
        self._getter = lambda index: available_items[index]
        self._setter = lambda element: available_items.index(element)
        self._printable = lambda index: f"{index} ({available_items[index].name})"
        self._frozen = True
        return self

    def build_list(self, items: Iterable[Any], value: Optional[Any] = None):
        if self._frozen:
            return self

        self._clear_all_properties()

        if not items:
            raise ValueError("The 'items' is empty")

        available_items = list(deepcopy(items))

        self._value = available_items.index(value) if value is not None else 0
        self._min_value = lambda: 0
        self._max_value = lambda: len(available_items) - 1
        self._decrease = lambda index: index - 1
        self._increase = lambda index: index + 1
        self._getter = lambda index: available_items[index]
        self._setter = lambda element: available_items.index(element)
        self._printable = lambda index: f"{index} ({available_items[index]})"
        self._frozen = True
        return self

    def build_latest_keycode(self, value=0):
        if self._frozen:
            return self

        self._clear_all_properties()

        def _keydown(keycode: int):
            self._value = keycode
            return True

        self._value = value
        self._keydown = _keydown
        self._frozen = True
        return self

    def build_select_roi(self, roi: Optional[RectInt] = None):
        if self._frozen:
            return self

        self._clear_all_properties()

        def _mouse(event: MouseEvent, mx: int, my: int, _):
            if event == MouseEvent.LBUTTON_DOWN:
                self._value = mx, my, self._value[2], self._value[3]
                self._kwargs["button_down"] = True
            if self._kwargs["button_down"]:
                if event == MouseEvent.MOUSE_MOVE:
                    self._value = self._value[0], self._value[1], mx, my
                elif event == MouseEvent.LBUTTON_UP:
                    self._value = self._value[0], self._value[1], mx, my
                    self._kwargs["button_down"] = False
            return True

        self._value = roi if roi else (0, 0, 0, 0)
        self._mouse = _mouse
        self._frozen = True
        self._kwargs["button_down"] = False
        return self

    def build_select_point(self, point: Optional[PointInt] = None):
        if self._frozen:
            return self

        self._clear_all_properties()

        def _mouse(event: MouseEvent, mx: int, my: int, _):
            if event.LBUTTON_DOWN:
                self._value = mx, my
            return True

        self._value = point
        self._mouse = _mouse
        self._frozen = True
        return self

    def build_select_points(self, points: Optional[Iterable[PointInt]] = None):
        if self._frozen:
            return self

        self._clear_all_properties()

        def _mouse(event: MouseEvent, mx: int, my: int, _):
            assert isinstance(self._value, list)
            if event == MouseEvent.LBUTTON_DOWN:
                self._value.append((mx, my))
            elif event == MouseEvent.MBUTTON_DOWN:
                self._value.pop()
            return True

        self._value = list(points) if points else list()
        self._mouse = _mouse
        self._frozen = True
        return self

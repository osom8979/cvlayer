# -*- coding: utf-8 -*-

from copy import deepcopy
from datetime import datetime
from io import StringIO
from types import TracebackType
from typing import Any, Dict, List, Literal, Optional, Tuple, Type
from weakref import ref

from numpy.typing import NDArray

from cvlayer.cv.mouse import EventFlags, MouseEvent
from cvlayer.layer.parameter import LayerParameter


class SkipError(ValueError):
    _default_msg = "An error occurred in the previous layer, so it cannot be executed"

    def __int__(self, *args):
        super().__init__(*args if args else self._default_msg)


class InvalidFrameError(ValueError):
    _default_msg = "The 'frame' property must be assigned"

    def __int__(self, *args):
        super().__init__(*args if args else self._default_msg)


class LayerBase:
    _params: Dict[str, LayerParameter]
    _error: Optional[BaseException]
    _frame: Optional[NDArray]

    def __init__(
        self,
        name: Optional[str] = None,
        prev: Optional[ref["LayerBase"]] = None,
        **params: LayerParameter,
    ):
        self._name = name if name else str()
        self._frame = None
        self._data = None
        self._params = params
        self._cursor = 0
        self._error = None
        self._begin = datetime.now()
        self._end = datetime.now()

        self._prev = prev

        self._keycode = 0
        self._mouse_event = MouseEvent.MOUSE_MOVE
        self._mouse_x = 0
        self._mouse_y = 0
        self._mouse_flags = 0

    @property
    def name(self):
        return self._name

    @property
    def prev_frame(self):
        assert self._prev is not None
        prev = self._prev()
        assert isinstance(prev, LayerBase)
        return prev.frame

    @property
    def prev_data(self):
        assert self._prev is not None
        prev = self._prev()
        assert isinstance(prev, LayerBase)
        return prev.data

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, value: NDArray) -> None:
        self._frame = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        self._data = value

    @property
    def mouse_event(self):
        return self._mouse_event

    @property
    def mouse_x(self):
        return self._mouse_x

    @property
    def mouse_y(self):
        return self._mouse_y

    @property
    def mouse_flags(self):
        return self._mouse_flags

    @property
    def keycode(self):
        return self._keycode

    def __str__(self):
        return self._name  # Important !!

    def __repr__(self) -> str:
        if self._name:
            return f"Layer[{self._name}]"
        else:
            return type(self).__name__

    def __getitem__(self, item: str) -> LayerParameter:
        return self.param(item)

    def __setitem__(self, key: str, value: LayerParameter) -> None:
        self._params[key] = value

    def __enter__(self):
        self._begin = datetime.now()
        self._error = None
        self._frame = None
        self._data = None
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[Literal[True]]:
        self._error = exc_val
        self._end = datetime.now()
        # If an exception is supplied, and the method wishes to suppress the exception
        # (i.e., prevent it from being propagated), it should return a true value
        if self._frame is None:
            if self._error is not None:
                raise self._error
            else:
                raise InvalidFrameError
        return None

    @property
    def keys(self) -> List[str]:
        return list(self._params.keys())

    @property
    def cursor_key(self) -> str:
        return list(self._params.keys())[self._cursor]

    @property
    def duration(self):
        return (self._end - self._begin).total_seconds()

    @property
    def has_error(self) -> bool:
        return self._error is not None

    @property
    def error(self) -> BaseException:
        assert self._error is not None
        return self._error

    def clear_error(self) -> None:
        self._error = None

    def param(self, key: str) -> LayerParameter:
        if key not in self._params:
            self._params[key] = LayerParameter()
        return self._params[key]

    def has(self, key: str) -> bool:
        return key in self._params

    def get(self, key: str) -> Any:
        return self._params[key].value

    def set(self, key: str, value: Any) -> None:
        self._params[key].value = value

    def decrease(self, key: str) -> None:
        self._params[key].do_decrease()

    def increase(self, key: str) -> None:
        self._params[key].do_increase()

    def set_cursor(self, index: int) -> None:
        if index < 0:
            self._cursor = 0
        elif index >= len(self._params):
            self._cursor = len(self._params) - 1
        else:
            self._cursor = index

    def prev_cursor(self) -> None:
        self.set_cursor(self._cursor - 1)

    def next_cursor(self) -> None:
        self.set_cursor(self._cursor + 1)

    def decrease_at_cursor(self) -> None:
        self._params[self.cursor_key].do_decrease()

    def increase_at_cursor(self) -> None:
        self._params[self.cursor_key].do_increase()

    def init_defaults(self) -> None:
        self._params = self.on_defaults()

    def as_help(self) -> str:
        buffer = StringIO()
        buffer.write(f"{repr(self)}")

        for key, param in self._params.items():
            prefix = ">" if self.cursor_key == key else " "
            readonly = "[RO] " if param.is_readonly else ""
            buffer.write(f"\n{prefix} {readonly}{key}: {param.as_printable_text()}")

        if self._error is not None:
            buffer.write(f"\n{type(self._error).__name__} {self._error}")

        return buffer.getvalue()

    def skip(self):
        self._begin = datetime.now()
        self._end = deepcopy(self._begin)
        self._error = SkipError()

    def run(self, frame: NDArray, data=None) -> Tuple[NDArray, Any]:
        self._begin = datetime.now()
        try:
            self._error = None
            self._frame, self._data = self.on_layer(frame, data)
        except BaseException as e:
            self._error = e
            raise e
        finally:
            self._end = datetime.now()
        return self._frame, self._data

    def on_defaults(self) -> Dict[str, LayerParameter]:
        assert self is not None
        return dict()

    def on_create(self) -> None:
        pass

    def on_destroy(self) -> None:
        pass

    def on_keydown(self, keycode: int) -> Optional[bool]:
        self._keycode = keycode

        if self._params:
            result = list()
            for param in self._params.values():
                if param.has_keydown:
                    result.append(param.call_keydown(keycode))
            if result:
                return any(result)

        return False

    def on_mouse(
        self,
        event: MouseEvent,
        x: int,
        y: int,
        flags: EventFlags,
    ) -> Optional[bool]:
        self._mouse_event = event
        self._mouse_x = x
        self._mouse_y = y
        self._mouse_flags = flags

        if self._params:
            result = list()
            for param in self._params.values():
                if param.has_mouse:
                    result.append(param.call_mouse(event, x, y, flags))
            if result:
                return any(result)

        return False

    def on_layer(self, frame: NDArray, data: Any) -> Tuple[NDArray, Any]:
        assert self
        return frame, data

# -*- coding: utf-8 -*-

from copy import deepcopy
from datetime import datetime
from io import StringIO
from types import TracebackType
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from numpy import zeros
from numpy.typing import NDArray

from cvlayer.cv.mouse import EventFlags, MouseEvent
from cvlayer.layers.base.layer_parameter import LayerParameter


class SkipError(ValueError):
    def __int__(self):
        msg = "An error occurred in the previous layer, so it cannot be executed"
        super().__init__(msg)


class LayerBase:
    _params: Dict[str, LayerParameter]
    _error: Optional[BaseException]

    def __init__(
        self,
        name: Optional[str] = None,
        **params: LayerParameter,
    ):
        self._name = name if name else str()
        self._frame = zeros([])
        self._data = None
        self._params = params
        self._cursor = 0
        self._error = None
        self._begin = datetime.now()
        self._end = datetime.now()

    @property
    def name(self):
        return self._name

    @property
    def frame(self):
        return self._frame

    @property
    def data(self):
        return self._data

    def __str__(self):
        return self._name

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        if self._name:
            return f"{cls_name}('{self._name}')"
        else:
            return cls_name

    def __getitem__(self, item: str) -> LayerParameter:
        return self.param(item)

    def __setitem__(self, key: str, value: LayerParameter) -> None:
        self._params[key] = value

    def __enter__(self):
        self._begin = datetime.now()
        self._error = None
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
        return None

    @property
    def keys(self) -> List[str]:
        return list(self._params.keys())

    @property
    def cursor_key(self) -> str:
        return list(self._params.keys())[self._cursor]

    @property
    def has_error(self) -> bool:
        return self._error is not None

    @property
    def duration(self):
        return (self._end - self._begin).total_seconds()

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

    def defaults(self) -> Dict[str, LayerParameter]:
        assert self is not None
        return dict()

    def init_defaults(self) -> None:
        self._params = self.defaults()

    def as_help(self) -> str:
        buffer = StringIO()
        buffer.write(f"{repr(self)}")

        for key, param in self._params.items():
            prefix = ">" if self.cursor_key == key else " "
            buffer.write(f"\n{prefix} {key}: {param.as_printable_text()}")

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

    def on_create(self) -> None:
        pass

    def on_destroy(self) -> None:
        pass

    def on_keydown(self, code: int) -> Optional[bool]:
        pass

    def on_mouse(
        self,
        event: MouseEvent,
        x: int,
        y: int,
        flags: EventFlags,
    ) -> Optional[bool]:
        pass

    def on_layer(self, frame: NDArray, data: Any) -> Tuple[NDArray, Any]:
        assert self
        return frame, data

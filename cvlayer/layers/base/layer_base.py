# -*- coding: utf-8 -*-

from abc import abstractmethod
from copy import deepcopy
from datetime import datetime
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple

from numpy import zeros
from numpy.typing import NDArray

from cvlayer.cv.mouse import EventFlags, MouseEvent
from cvlayer.layers.base.layer_parameter import LayerParameter


class SkipError(ValueError):
    def __int__(self):
        msg = "An error occurred in the previous layer, so it cannot be executed"
        super().__init__(msg)


class LayerBase:
    _prev_frame: NDArray
    _next_frame: NDArray

    _prev_data: Any
    _next_data: Any

    _params: Dict[str, LayerParameter]
    _param_cursor: int

    _error: Optional[BaseException]

    _duration: float

    def __init__(self, name: Optional[str] = None, deep_copy_params=True):
        self._name = name
        self._deep_copy_params = deep_copy_params

        self._prev_frame = zeros([])
        self._next_frame = zeros([])

        self._prev_data = None
        self._next_data = None

        self._params = dict()
        self._param_cursor = 0
        self._error = None

        self._duration = 0.0

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        if self._name:
            return f"{cls_name}<{self._name}>"
        else:
            return cls_name

    @property
    def has_name(self) -> bool:
        return bool(self._name)

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        else:
            return str()

    @property
    def prev_frame(self) -> NDArray:
        return self._prev_frame

    @property
    def next_frame(self) -> NDArray:
        return self._next_frame

    @property
    def prev_data(self) -> Any:
        return self._prev_data

    @property
    def next_data(self) -> Any:
        return self._next_data

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def param_cursor_key(self) -> str:
        return list(self._params.keys())[self._param_cursor]

    @property
    def keys(self) -> List[str]:
        return list(self._params.keys())

    @property
    def has_error(self) -> bool:
        return self._error is not None

    def has(self, key: str) -> bool:
        return key in self._params

    def get(self, key: str) -> Any:
        return self._params[key].value

    def set(self, key: str, value: Any) -> None:
        self._params[key].value = value

    def decrease(self, key: str) -> None:
        self._params[key].decrease()

    def increase(self, key: str) -> None:
        self._params[key].increase()

    def change_param_cursor(self, index: int) -> None:
        if index < 0:
            self._param_cursor = 0
        elif index >= len(self._params):
            self._param_cursor = len(self._params) - 1
        else:
            self._param_cursor = index

    def prev_param_cursor(self) -> None:
        self.change_param_cursor(self._param_cursor - 1)

    def next_param_cursor(self) -> None:
        self.change_param_cursor(self._param_cursor + 1)

    def decrease_at_param_cursor(self) -> None:
        self._params[self.param_cursor_key].decrease()

    def increase_at_param_cursor(self) -> None:
        self._params[self.param_cursor_key].increase()

    def init_params(self) -> Dict[str, LayerParameter]:
        assert self is not None
        return dict()

    def to_help(self) -> str:
        buffer = StringIO()
        buffer.write(f"{repr(self)}")

        for key, param in self._params.items():
            prefix = ">" if self.param_cursor_key == key else " "
            buffer.write(f"\n{prefix} {key}: {param.printable()}")

        if self._error is not None:
            buffer.write(f"\n{type(self._error).__name__} {self._error}")

        return buffer.getvalue()

    def skip(self):
        self._error = SkipError()
        self._duration = 0.0

    def run(self, frame: NDArray, data: Any) -> Tuple[NDArray, Any]:
        if self._deep_copy_params:
            self._prev_frame = frame.copy()
            self._prev_frame.setflags(write=False)
            self._prev_data = deepcopy(data)
        else:
            self._prev_frame = frame
            self._prev_data = data

        running_frame = frame.copy() if self._deep_copy_params else frame

        next_frame: NDArray
        next_data: Any

        begin = datetime.now()
        try:
            self._error = None
            next_frame, next_data = self.on_layer(running_frame, data)
        except BaseException as e:
            self._error = e
            raise e
        finally:
            self._duration = (datetime.now() - begin).total_seconds()

        if self._deep_copy_params:
            self._next_frame = next_frame.copy()
            self._next_frame.setflags(write=False)
            self._next_data = deepcopy(next_data)
        else:
            self._next_frame = next_frame
            self._next_data = next_data

        return next_frame, next_data

    def on_create(self) -> None:
        assert self is not None

    def on_destroy(self) -> None:
        assert self is not None

    def on_keydown(self, code: int) -> bool:
        assert self is not None
        assert isinstance(code, int)
        return False  # If `True`, the event is consumed.

    def on_mouse(self, event: MouseEvent, x: int, y: int, flags: EventFlags) -> None:
        pass

    @abstractmethod
    def on_layer(self, frame: NDArray, data: Any) -> Tuple[NDArray, Any]:
        raise NotImplementedError


OnLayerCallable = Callable[[NDArray, Any], Tuple[NDArray, Any]]

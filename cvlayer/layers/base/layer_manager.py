# -*- coding: utf-8 -*-

from copy import deepcopy
from functools import reduce
from logging import getLogger
from typing import Any, Dict, Final, List, Optional, Tuple, Union
from weakref import ref

from numpy.typing import NDArray

from cvlayer.cv.mouse import EventFlags, MouseEvent
from cvlayer.layers.base.layer_base import LayerBase, SkipError

LAST_LAYER_INDEX: Final[int] = -1


class LayerManager:
    _layers: List[LayerBase]
    _name2index: Dict[str, int]

    def __init__(self, cursor=LAST_LAYER_INDEX, logger_name: Optional[str] = None):
        self._cursor = cursor
        self._layers = list()
        self._name2index = dict()
        self._logger = getLogger(logger_name)

        self._pseudo_first = LayerBase("__pseudo_first__", None)

    def __getitem__(self, key: Any) -> LayerBase:
        if not self.has_layer(key):
            self.append_layer(str(key))
        return self.get_layer(key)

    def __setitem__(self, key: str, value: LayerBase) -> None:
        raise NotImplementedError("Unsupported __setitem__ method")

    @property
    def cursor(self):
        return self._cursor

    @property
    def logger(self):
        return self._logger

    @property
    def current_layer(self) -> LayerBase:
        return self._layers[self._cursor]

    @property
    def first_layer(self) -> LayerBase:
        return self._layers[0]

    @property
    def last_layer(self) -> LayerBase:
        return self._layers[LAST_LAYER_INDEX]

    @property
    def number_of_layers(self) -> int:
        return len(self._layers)

    @property
    def is_cursor_at_first(self) -> bool:
        return self._cursor == 0

    @property
    def is_cursor_at_last(self) -> bool:
        return self._cursor == LAST_LAYER_INDEX

    @property
    def total_duration(self) -> float:
        if self.number_of_layers >= 1:
            durations = map(lambda x: x.duration, self._layers)
            return float(reduce(lambda x, y: x + y, durations))
        else:
            return 0.0

    def append_layer(self, name: str) -> None:
        if name in self._name2index:
            raise KeyError(f"A layer with the same name already exists: '{name}'")

        prev = self._layers[-1] if self._layers else self._pseudo_first
        layer = LayerBase(name, ref(prev))

        self._layers.append(layer)
        self._name2index[name] = len(self._layers) - 1

    def has_layer(self, key: Any) -> bool:
        return str(key) in self._name2index

    def get_layer_index(self, key: Any) -> int:
        return self._name2index[str(key)]

    def get_layer(self, key: Any) -> LayerBase:
        return self._layers[self._name2index[str(key)]]

    def get_layer_by_index(self, index: int) -> LayerBase:
        return self._layers[index]

    def get_layer_by_name(self, name: str) -> LayerBase:
        return self._layers[self.get_layer_index(name)]

    def get_layer_frame(self, key: Union[int, str]) -> NDArray:
        return self.get_layer(key).frame

    def get_layer_data(self, key: Union[int, str]) -> Any:
        return self.get_layer(key).data

    def has_layer_param(self, layer_key: Union[int, str], param_key: str) -> bool:
        return self.get_layer(layer_key).has(param_key)

    def get_layer_param(self, layer_key: Union[int, str], param_key: str) -> Any:
        return self.get_layer(layer_key).get(param_key)

    def set_layer_param(
        self,
        layer_key: Union[int, str],
        param_key: str,
        value: Any,
    ) -> None:
        return self.get_layer(layer_key).set(param_key, value)

    def set_cursor(self, cursor: int) -> None:
        if cursor == LAST_LAYER_INDEX or cursor == len(self._layers):
            self._cursor = LAST_LAYER_INDEX
            return

        if not self._layers:
            raise IndexError("LayerBase does not exist")

        if 0 <= cursor < len(self._layers):
            self._cursor = cursor
        else:
            raise IndexError(f"LayerBase index out of range: {cursor}")

    def move_last_layer(self) -> None:
        self._cursor = LAST_LAYER_INDEX

    def move_prev_layer(self) -> None:
        if not self._layers:
            raise IndexError("LayerBase does not exist")

        if self._cursor == LAST_LAYER_INDEX:
            self._cursor = len(self._layers) - 1
            return

        prev_index = self._cursor - 1
        self._cursor = prev_index if prev_index >= 0 else LAST_LAYER_INDEX

    def move_next_layer(self) -> None:
        if not self._layers:
            raise IndexError("LayerBase does not exist")

        if self._cursor == LAST_LAYER_INDEX:
            self._cursor = 0
            return

        next_index = self._cursor + 1
        max_index = len(self._layers)
        self._cursor = next_index if next_index < max_index else LAST_LAYER_INDEX

    def logging_current_param(self) -> None:
        if self.is_cursor_at_last:
            self._logger.info("No layer have been selected")
            return

        current_layer = self.current_layer
        layer_type = type(current_layer).__name__

        key = current_layer.cursor_key
        value = current_layer.get(key)
        self._logger.info(f"[{layer_type}] '{key}' parameter value: {value}")

    def logging_current_layer(self) -> None:
        index = self._cursor
        max_index = len(self._layers) - 1
        name = type(self._layers[index]).__name__
        self._logger.info(f"Change layer ({index}/{max_index}) '{name}'")

    def update_first_frame_and_data(self, frame: NDArray, data=None) -> None:
        self._pseudo_first.frame = frame
        self._pseudo_first.data = data

    def run(self, frame: NDArray, data=None, use_deepcopy=False) -> Tuple[NDArray, Any]:
        if not self._layers:
            return frame, data

        prev_layer: Optional[LayerBase] = None
        next_frame: NDArray = frame
        next_data: Any = data

        for layer_index, layer in enumerate(self._layers):
            assert isinstance(layer, LayerBase)
            if prev_layer is not None:
                if prev_layer.has_error:
                    layer.skip()
                    continue

            if use_deepcopy:
                next_frame = next_frame.copy()
                next_frame.setflags(write=False)
                next_data = deepcopy(data)

            try:
                next_frame, next_data = layer.run(next_frame, next_data)
            except SkipError:
                continue
            except BaseException as e:
                self._logger.exception(e)
            finally:
                prev_layer = layer

        return next_frame, next_data

    def on_create(self, init_defaults=True) -> None:
        for layer in self._layers:
            if init_defaults:
                layer.init_defaults()
            layer.on_create()

    def on_destroy(self) -> None:
        for layer in self._layers:
            layer.on_destroy()

    def on_keydown(self, keycode: int) -> Optional[bool]:
        if self.is_cursor_at_last:
            return False
        else:
            return self.current_layer.on_keydown(keycode)

    def on_mouse(
        self,
        event: MouseEvent,
        x: int,
        y: int,
        flags: EventFlags,
    ) -> Optional[bool]:
        if self.is_cursor_at_last:
            return False
        else:
            return self.current_layer.on_mouse(event, x, y, flags)

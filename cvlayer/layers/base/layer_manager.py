# -*- coding: utf-8 -*-

from copy import deepcopy
from logging import getLogger
from typing import Any, Dict, Final, List, Optional, Tuple

from numpy.typing import NDArray

from cvlayer.cv.mouse import EventFlags, MouseEvent
from cvlayer.layers.base.layer_base import LayerBase, SkipError

LAST_LAYER_INDEX: Final[int] = -1


class LayerManager:
    layers: List[LayerBase]
    name2index: Dict[str, int]

    def __init__(
        self,
        name: Optional[str] = None,
        index=LAST_LAYER_INDEX,
        logger_name: Optional[str] = None,
    ):
        self.name = name if name else str()
        self.index = index
        self.layers = list()
        self.name2index = dict()
        self.logger = getLogger(logger_name)

    def __getitem__(self, item: str) -> LayerBase:
        if not self.has_layer_by_name(item):
            self.append_layer(LayerBase(item))
        return self.get_layer_by_name(item)

    def __setitem__(self, key: str, value: LayerBase) -> None:
        self.append_layer(value)

    @property
    def current_layer(self) -> LayerBase:
        return self.layers[self.index]

    @property
    def first_layer(self) -> LayerBase:
        return self.layers[0]

    @property
    def last_layer(self) -> LayerBase:
        return self.layers[LAST_LAYER_INDEX]

    @property
    def number_of_layers(self) -> int:
        return len(self.layers)

    @property
    def is_first_layer(self) -> bool:
        return self.index == 0

    @property
    def is_last_layer(self) -> bool:
        return self.index == LAST_LAYER_INDEX

    def append_layer(self, layer: LayerBase) -> None:
        if layer.name in self.name2index:
            raise KeyError(f"A layer with the same name already exists: '{layer.name}'")

        self.layers.append(layer)
        if layer.name:
            self.name2index[layer.name] = len(self.layers) - 1

    def has_layer_by_name(self, name: str) -> bool:
        return name in self.name2index

    def get_layer_index_by_name(self, name: str) -> int:
        return self.name2index[name]

    def get_layer_by_name(self, name: str) -> LayerBase:
        return self.layers[self.get_layer_index_by_name(name)]

    def get_layer_frame(self, index: int) -> NDArray:
        return self.layers[index].frame

    def get_layer_data(self, index: int) -> Any:
        return self.layers[index].data

    def has_layer_param(self, index: int, key: str) -> bool:
        return self.layers[index].has(key)

    def get_layer_param(self, index: int, key: str) -> Any:
        return self.layers[index].get(key)

    def set_layer_param(self, index: int, key: str, value: Any) -> None:
        self.layers[index].set(key, value)

    def set_index(self, index: int) -> None:
        if index == LAST_LAYER_INDEX or index == len(self.layers):
            self.index = LAST_LAYER_INDEX
            return

        if not self.layers:
            raise IndexError("LayerBase does not exist")

        if 0 <= index < len(self.layers):
            self.index = index
        else:
            raise IndexError(f"LayerBase index out of range: {index}")

    def set_last_index(self) -> None:
        self.index = LAST_LAYER_INDEX

    def prev_layer(self) -> None:
        if not self.layers:
            raise IndexError("LayerBase does not exist")

        if self.index == LAST_LAYER_INDEX:
            self.index = len(self.layers) - 1
            return

        prev_index = self.index - 1
        self.index = prev_index if prev_index >= 0 else LAST_LAYER_INDEX

    def next_layer(self) -> None:
        if not self.layers:
            raise IndexError("LayerBase does not exist")

        if self.index == LAST_LAYER_INDEX:
            self.index = 0
            return

        next_index = self.index + 1
        max_index = len(self.layers)
        self.index = next_index if next_index < max_index else LAST_LAYER_INDEX

    def logging_current_param(self) -> None:
        if self.is_last_layer:
            self.logger.info("No layer have been selected")
            return

        current_layer = self.current_layer
        layer_type = type(current_layer).__name__

        key = current_layer.cursor_key
        value = current_layer.get(key)
        self.logger.info(f"[{layer_type}] '{key}' parameter value: {value}")

    def logging_current_layer(self) -> None:
        index = self.index
        max_index = len(self.layers) - 1
        name = type(self.layers[index]).__name__
        self.logger.info(f"Change layer ({index}/{max_index}) '{name}'")

    def run(self, frame: NDArray, data=None, use_deepcopy=False) -> Tuple[NDArray, Any]:
        if not self.layers:
            return frame, data

        prev_layer: Optional[LayerBase] = None
        next_frame: NDArray = frame
        next_data: Any = data

        for layer_index, layer in enumerate(self.layers):
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
                self.logger.exception(e)
            finally:
                prev_layer = layer

        return next_frame, next_data

    def on_create(self, init_defaults=True) -> None:
        for layer in self.layers:
            if init_defaults:
                layer.init_defaults()
            layer.on_create()

    def on_destroy(self) -> None:
        for layer in self.layers:
            layer.on_destroy()

    def on_keydown(self, keycode: int) -> Optional[bool]:
        if self.is_last_layer:
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
        if self.is_last_layer:
            return False
        else:
            return self.current_layer.on_mouse(event, x, y, flags)

# -*- coding: utf-8 -*-

from logging import Logger, getLogger
from typing import Any, Dict, Final, List, Optional

from numpy.typing import NDArray

from cvlayer.cv.mouse import EventFlags, MouseEvent
from cvlayer.layers.base.layer_base import LayerBase, SkipError

LAST_LAYER_INDEX: Final[int] = -1


class LayerManager:
    _layers: List[LayerBase]
    _name_to_index: Dict[str, int]

    def __init__(
        self,
        layer_index=LAST_LAYER_INDEX,
        logger_name: Optional[str] = None,
        deep_copy_params=True,
    ):
        self._layer_index = layer_index
        self._logger_name = logger_name
        self._deep_copy_params = deep_copy_params

        self._layers = list()
        self._name_to_index = dict()

    @property
    def logger(self) -> Logger:
        return getLogger(self._logger_name)

    @property
    def deep_copy_params(self) -> bool:
        return self._deep_copy_params

    @property
    def layers(self) -> List[LayerBase]:
        assert isinstance(self._layers, list)
        return self._layers

    @property
    def layer_index(self) -> int:
        return self._layer_index

    @property
    def current_layer(self) -> LayerBase:
        return self._layers[self._layer_index]

    @property
    def length_layers(self) -> int:
        return len(self._layers)

    @property
    def is_last_layer(self) -> bool:
        return self._layer_index == LAST_LAYER_INDEX

    def append_layer(self, layer: LayerBase) -> None:
        self._layers.append(layer)
        if layer.has_name:
            self._name_to_index[layer.name] = len(self._layers) - 1

    def get_layer_index_by_name(self, name: str) -> int:
        return self._name_to_index[name]

    def get_layer_by_name(self, name: str) -> LayerBase:
        return self._layers[self.get_layer_index_by_name(name)]

    def get_layer_prev_frame(self, index: int) -> NDArray:
        return self._layers[index].prev_frame

    def get_layer_prev_data(self, index: int) -> Any:
        return self._layers[index].prev_data

    def get_layer_next_frame(self, index: int) -> NDArray:
        return self._layers[index].next_frame

    def get_layer_next_data(self, index: int) -> Any:
        return self._layers[index].next_data

    def has_layer_param(self, index: int, key: str) -> bool:
        return self._layers[index].has(key)

    def get_layer_param(self, index: int, key: str) -> Any:
        return self._layers[index].get(key)

    def set_layer_param(self, index: int, key: str, value: Any) -> None:
        self._layers[index].set(key, value)

    def change_layer(self, index: int) -> None:
        if index == LAST_LAYER_INDEX or index == len(self._layers):
            self._layer_index = LAST_LAYER_INDEX
            return

        if not self._layers:
            raise IndexError("LayerBase does not exist")

        if 0 <= index < len(self._layers):
            self._layer_index = index
        else:
            raise IndexError(f"LayerBase index out of range: {index}")

    def change_last_layer(self) -> None:
        self._layer_index = LAST_LAYER_INDEX

    def change_prev_layer(self) -> None:
        if not self._layers:
            raise IndexError("LayerBase does not exist")

        if self._layer_index == LAST_LAYER_INDEX:
            self._layer_index = len(self._layers) - 1
            return

        prev_index = self._layer_index - 1
        self._layer_index = prev_index if prev_index >= 0 else LAST_LAYER_INDEX

    def change_next_layer(self) -> None:
        if not self._layers:
            raise IndexError("LayerBase does not exist")

        if self._layer_index == LAST_LAYER_INDEX:
            self._layer_index = 0
            return

        next_index = self._layer_index + 1
        max_index = len(self._layers)
        self._layer_index = next_index if next_index < max_index else LAST_LAYER_INDEX

    def logging_current_param_cursor(self) -> None:
        if self.is_last_layer:
            self.logger.info("No layer have been selected")
            return

        current_layer = self.current_layer
        layer_type = type(current_layer).__name__

        key = current_layer.param_cursor_key
        value = current_layer.get(key)
        self.logger.info(f"[{layer_type}] '{key}' parameter value: {value}")

    def logging_current_layer_information(self) -> None:
        index = self._layer_index
        max_index = len(self._layers) - 1
        name = type(self._layers[index]).__name__
        self.logger.info(f"Change layer ({index}/{max_index}) '{name}'")

    def call_init_params_with_layers(self) -> None:
        for layer in self._layers:
            layer._params = layer.init_params()

    def call_on_frame_with_layers(self, frame: NDArray) -> NDArray:
        if not self._layers:
            return frame

        prev_layer: Optional[LayerBase] = None
        next_frame: NDArray = frame
        next_data: Any = None

        for layer_index, layer in enumerate(self._layers):
            assert isinstance(layer, LayerBase)
            if prev_layer is not None:
                if prev_layer.has_error:
                    layer.skip()
                    continue

            try:
                next_frame, next_data = layer.run(next_frame, next_data)
            except SkipError:
                continue
            except BaseException as e:
                self.logger.exception(e)
            finally:
                prev_layer = layer

        return next_frame

    def call_on_keydown_with_current_layer(self, keycode: int) -> bool:
        if self.is_last_layer:
            return False  # If `True`, the event is consumed.
        else:
            return self.current_layer.on_keydown(keycode)

    def call_on_mouse_with_current_layer(
        self, event: MouseEvent, x: int, y: int, flags: EventFlags
    ) -> None:
        if not self.is_last_layer:
            return self.current_layer.on_mouse(event, x, y, flags)

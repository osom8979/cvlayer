# -*- coding: utf-8 -*-

from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

from numpy import zeros
from numpy.typing import NDArray

from cvlayer.cv.mouse import EventFlags, MouseEvent
from cvlayer.layers.base.layer_parameter import LayerParameter


class SkipError(ValueError):
    def __int__(self):
        msg = "An error occurred in the previous layer, so it cannot be executed"
        super().__init__(msg)


class LayerBase:
    name: str
    frame: NDArray
    data: Any
    params: Dict[str, LayerParameter]
    cursor: int
    error: Optional[BaseException]
    duration: float
    use_deepcopy: bool

    def __init__(
        self,
        name: Optional[str] = None,
        **params: LayerParameter,
    ):
        self.name = name if name else str()
        self.frame = zeros([])
        self.data = None
        self.params = params
        self.cursor = 0
        self.error = None
        self.duration = 0.0
        self.use_deepcopy = False

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        if self.name:
            return f"{cls_name}<{self.name}>"
        else:
            return cls_name

    @property
    def keys(self) -> List[str]:
        return list(self.params.keys())

    @property
    def cursor_key(self) -> str:
        return list(self.params.keys())[self.cursor]

    @property
    def has_error(self) -> bool:
        return self.error is not None

    def has(self, key: str) -> bool:
        return key in self.params

    def get(self, key: str) -> Any:
        return self.params[key].value

    def set(self, key: str, value: Any) -> None:
        self.params[key].value = value

    def decrease(self, key: str) -> None:
        self.params[key].decrease()

    def increase(self, key: str) -> None:
        self.params[key].increase()

    def set_cursor(self, index: int) -> None:
        if index < 0:
            self.cursor = 0
        elif index >= len(self.params):
            self.cursor = len(self.params) - 1
        else:
            self.cursor = index

    def prev_cursor(self) -> None:
        self.set_cursor(self.cursor - 1)

    def next_cursor(self) -> None:
        self.set_cursor(self.cursor + 1)

    def decrease_at_cursor(self) -> None:
        self.params[self.cursor_key].decrease()

    def increase_at_cursor(self) -> None:
        self.params[self.cursor_key].increase()

    def defaults(self) -> Dict[str, LayerParameter]:
        assert self is not None
        return dict()

    def as_help(self) -> str:
        buffer = StringIO()
        buffer.write(f"{repr(self)}")

        for key, param in self.params.items():
            prefix = ">" if self.cursor_key == key else " "
            buffer.write(f"\n{prefix} {key}: {param.printable()}")

        if self.error is not None:
            buffer.write(f"\n{type(self.error).__name__} {self.error}")

        return buffer.getvalue()

    def skip(self):
        self.error = SkipError()
        self.duration = 0.0

    def run(self, frame: NDArray, data=None) -> Tuple[NDArray, Any]:
        begin = datetime.now()
        try:
            self.error = None
            self.frame, self.data = self.on_layer(frame, data)
        except BaseException as e:
            self.error = e
            raise e
        finally:
            self.duration = (datetime.now() - begin).total_seconds()

        return self.frame, self.data

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

# -*- coding: utf-8 -*-

from typing import Any

from cvlayer.layer.base import LayerBase
from cvlayer.layer.manager.interface import LayerManagerInterface
from cvlayer.typing import override


class LayerManagerMixinBase(LayerManagerInterface):
    @override
    def layer(self, key: Any) -> LayerBase:
        raise NotImplementedError

    @override
    def set_roi(self, roi: Any) -> None:
        raise NotImplementedError

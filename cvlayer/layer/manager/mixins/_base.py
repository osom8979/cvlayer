# -*- coding: utf-8 -*-

from typing import Any

from overrides import override

from cvlayer.layer.base import LayerBase
from cvlayer.layer.manager.interface import LayerManagerInterface


class _LayerManagerMixinBase(LayerManagerInterface):
    @override
    def layer(self, key: Any) -> LayerBase:
        raise NotImplementedError

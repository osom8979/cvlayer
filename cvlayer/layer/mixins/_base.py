# -*- coding: utf-8 -*-

from typing import Any

from cvlayer.layer.layer_base import LayerBase


class _LayerManagerMixinBase:
    def _layer(self, key: Any) -> LayerBase:
        return getattr(self, "layer")(key)

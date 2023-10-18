# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from typing import Any

from cvlayer.layer.base import LayerBase


class LayerManagerInterface(metaclass=ABCMeta):
    @abstractmethod
    def layer(self, key: Any) -> LayerBase:
        raise NotImplementedError

    @abstractmethod
    def set_roi(self, roi: Any) -> None:
        raise NotImplementedError

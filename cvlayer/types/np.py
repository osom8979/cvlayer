# -*- coding: utf-8 -*-

from typing import Literal, Tuple

from numpy import uint8
from numpy.typing import NDArray

Image = NDArray[uint8]

_Height = int
_Width = int
_Channels = Literal[3]

ImageShape = Tuple[_Height, _Width, _Channels]

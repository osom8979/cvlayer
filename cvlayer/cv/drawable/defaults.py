# -*- coding: utf-8 -*-

from typing import Final, Tuple, Union

from cvlayer.cv.types.color import ColorLike
from cvlayer.cv.types.font_face import HersheyFont
from cvlayer.cv.types.line_type import LineType, LineTypeLike
from cvlayer.cv.types.text_origin import DEFAULT_TEXT_ORIGIN as _DEFAULT_TEXT_ORIGIN
from cvlayer.cv.types.text_origin import TextOriginLike
from cvlayer.palette.basic import BLACK, WHITE

DEFAULT_RADIUS: Final[int] = 4
DEFAULT_THICKNESS: Final[int] = 2
DEFAULT_COLOR: Final[ColorLike] = BLACK
DEFAULT_LINE_TYPE: Final[LineTypeLike] = LineType.AA
DEFAULT_SHIFT: Final[int] = 0

DEFAULT_FONT_COLOR: Final[ColorLike] = BLACK
DEFAULT_FONT_FACE: Final[Union[HersheyFont, int]] = HersheyFont.SIMPLEX
DEFAULT_FONT_SCALE: Final[float] = 1.0
DEFAULT_TEXT_ORIGIN: Final[TextOriginLike] = _DEFAULT_TEXT_ORIGIN

DEFAULT_FONT_OUTLINE_COLOR: Final[ColorLike] = WHITE
DEFAULT_FONT_OUTLINE_THICKNESS: Final[int] = 4

MULTILINE_COLOR: Final[ColorLike] = (220, 220, 220)
MULTILINE_LINEFEED: Final[str] = "\n"
MULTILINE_LINE_SPACING: Final[int] = 4
MULTILINE_BACKGROUND_COLOR: Final[ColorLike] = BLACK
MULTILINE_BACKGROUND_ALPHA: Final[float] = 0.4
MULTILINE_BOX_MARGIN: Final[int] = 8
MULTILINE_BOX_ANCHOR_X: Final[float] = 0.0
MULTILINE_BOX_ANCHOR_Y: Final[float] = 0.0
MULTILINE_BOX_ANCHOR: Final[Tuple[float, float]] = (
    MULTILINE_BOX_ANCHOR_X,
    MULTILINE_BOX_ANCHOR_Y,
)

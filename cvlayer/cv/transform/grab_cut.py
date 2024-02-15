# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Dict, Final, NamedTuple, Optional, Union

import cv2
from numpy import int8, uint8
from numpy.typing import NDArray

from cvlayer.typing import RectI

GC_INIT_WITH_RECT: Final[int] = cv2.GC_INIT_WITH_RECT
GC_INIT_WITH_MASK: Final[int] = cv2.GC_INIT_WITH_MASK
GC_EVAL: Final[int] = cv2.GC_EVAL
GC_EVAL_FREEZE_MODEL: Final[int] = cv2.GC_EVAL_FREEZE_MODEL

GC_BGD: Final[int] = cv2.GC_BGD
GC_FGD: Final[int] = cv2.GC_FGD
GC_PR_BGD: Final[int] = cv2.GC_PR_BGD
GC_PR_FGD: Final[int] = cv2.GC_PR_FGD

assert GC_BGD == 0
assert GC_FGD == 1
assert GC_PR_BGD == 2
assert GC_PR_FGD == 3


@unique
class GrabCutClass(Enum):
    BGD = GC_BGD
    """an obvious background pixels"""

    FGD = GC_FGD
    """an obvious foreground (object) pixel"""

    PR_BGD = GC_PR_BGD
    """a possible background pixel"""

    PR_FGD = GC_PR_FGD
    """a possible foreground pixel"""


GrabCutClassLike = Union[GrabCutClass, str, int]

GRAB_CUT_CLASS_MAP: Final[Dict[str, int]] = {
    # str to int names
    "0": GC_BGD,
    "1": GC_FGD,
    "2": GC_PR_BGD,
    "3": GC_PR_FGD,
    # GrabCutClass enum names
    "BGD": GC_BGD,
    "FGD": GC_FGD,
    "PR_BGD": GC_PR_BGD,
    "PR_FGD": GC_PR_FGD,
    # cv2 symbol full names
    "GC_BGD": GC_BGD,
    "GC_FGD": GC_FGD,
    "GC_PR_BGD": GC_PR_BGD,
    "GC_PR_FGD": GC_PR_FGD,
}


def normalize_grab_cut_class(value: GrabCutClassLike) -> int:
    if isinstance(value, GrabCutClass):
        return value.value
    elif isinstance(value, int):
        return value
    else:
        raise TypeError(f"Unsupported value type: {type(value).__name__}")


@unique
class GrabCutMode(Enum):
    """GrabCut algorithm flags"""

    INIT_WITH_RECT = GC_INIT_WITH_RECT
    """
    The function initializes the state and the mask using the provided rectangle.
    After that it runs iterCount iterations of the algorithm.
    """

    INIT_WITH_MASK = GC_INIT_WITH_MASK
    """
    The function initializes the state using the provided mask.
    Note that GC_INIT_WITH_RECT and GC_INIT_WITH_MASK can be combined.
    Then, all the pixels outside of the ROI are automatically initialized with GC_BGD.
    """

    EVAL = GC_EVAL
    """
    The value means that the algorithm should just resume.
    """

    EVAL_FREEZE_MODEL = GC_EVAL_FREEZE_MODEL
    """
    The value means that the algorithm should just run the grabCut algorithm
    (a single iteration) with the fixed model.
    """


GrabCutModeLike = Union[GrabCutMode, str, int]

DEFAULT_GRAB_CUT_MODE: Final[GrabCutModeLike] = GC_EVAL
GRAB_CUT_MODE_MAP: Final[Dict[str, int]] = {
    # GrabCutMode enum names
    "INIT_WITH_RECT": GC_INIT_WITH_RECT,
    "INIT_WITH_MASK": GC_INIT_WITH_MASK,
    "EVAL": GC_EVAL,
    "EVAL_FREEZE_MODEL": GC_EVAL_FREEZE_MODEL,
    # cv2 symbol full names
    "GC_INIT_WITH_RECT": GC_INIT_WITH_RECT,
    "GC_INIT_WITH_MASK": GC_INIT_WITH_MASK,
    "GC_EVAL": GC_EVAL,
    "GC_EVAL_FREEZE_MODEL": GC_EVAL_FREEZE_MODEL,
}


def normalize_grab_cut_mode(mode: Optional[GrabCutModeLike]) -> int:
    if mode is None:
        assert isinstance(DEFAULT_GRAB_CUT_MODE, int)
        return DEFAULT_GRAB_CUT_MODE

    if isinstance(mode, GrabCutMode):
        return mode.value
    elif isinstance(mode, int):
        return mode
    else:
        raise TypeError(f"Unsupported mode type: {type(mode).__name__}")


class GrabCutResult(NamedTuple):
    mask: NDArray
    background_model: NDArray
    foreground_model: NDArray


def grab_cut(
    image: NDArray,
    mask: NDArray,
    rect: Optional[RectI],
    background_model: NDArray,
    foreground_model: NDArray,
    iterations: int,
    mode=DEFAULT_GRAB_CUT_MODE,
) -> GrabCutResult:
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"The image argument must be 3-channels: {image.shape}")
    if image.dtype not in (int8, uint8):
        raise TypeError(f"The image argument must be 8-bits: {image.dtype}")

    if len(mask.shape) != 2:
        raise ValueError(f"The mask argument must be single-channels: {mask.shape}")
    if mask.dtype not in (int8, uint8):
        raise TypeError(f"The mask argument must be 8-bits: {mask.dtype}")

    _mode = normalize_grab_cut_mode(mode)

    if _mode == GC_INIT_WITH_RECT and rect is None:
        raise ValueError("The rect argument is required when mode == GC_INIT_WITH_RECT")

    result = cv2.grabCut(
        img=image,
        mask=mask,
        rect=rect,  # type: ignore[arg-type]
        bgdModel=background_model,
        fgdModel=foreground_model,
        iterCount=iterations,
        mode=_mode,
    )
    ret_mask, ret_bg, ret_fg = result
    return GrabCutResult(ret_mask, ret_bg, ret_fg)


def grab_cut_rect(
    image: NDArray,
    mask: NDArray,
    rect: RectI,
    background_model: NDArray,
    foreground_model: NDArray,
    iterations: int,
) -> GrabCutResult:
    return grab_cut(
        image,
        mask,
        rect,
        background_model,
        foreground_model,
        iterations,
        GC_INIT_WITH_RECT,
    )


def fill_grab_cut_class(image: NDArray, value: GrabCutClassLike) -> None:
    image[:] = normalize_grab_cut_class(value)


def fill_grab_cut_obvious_background(image: NDArray) -> None:
    fill_grab_cut_class(image, GC_BGD)


def fill_grab_cut_obvious_foreground(image: NDArray) -> None:
    fill_grab_cut_class(image, GC_FGD)


def fill_grab_cut_possible_background(image: NDArray) -> None:
    fill_grab_cut_class(image, GC_PR_BGD)


def fill_grab_cut_possible_foreground(image: NDArray) -> None:
    fill_grab_cut_class(image, GC_PR_FGD)


class CvlTransformGrabCut:
    @staticmethod
    def cvl_normalize_grab_cut_class(value: GrabCutClassLike):
        return normalize_grab_cut_class(value)

    @staticmethod
    def cvl_normalize_grab_cut_mode(mode: Optional[GrabCutModeLike]):
        return normalize_grab_cut_mode(mode)

    @staticmethod
    def cvl_grab_cut(
        image: NDArray,
        mask: NDArray,
        rect: Optional[RectI],
        background_model: NDArray,
        foreground_model: NDArray,
        iterations: int,
        mode=DEFAULT_GRAB_CUT_MODE,
    ):
        return grab_cut(
            image,
            mask,
            rect,
            background_model,
            foreground_model,
            iterations,
            mode,
        )

    @staticmethod
    def cvl_grab_cut_rect(
        image: NDArray,
        mask: NDArray,
        rect: RectI,
        background_model: NDArray,
        foreground_model: NDArray,
        iterations: int,
    ):
        return grab_cut_rect(
            image,
            mask,
            rect,
            background_model,
            foreground_model,
            iterations,
        )

    @staticmethod
    def cvl_fill_grab_cut_class(image: NDArray, value: GrabCutClassLike):
        return fill_grab_cut_class(image, value)

    @staticmethod
    def cvl_fill_grab_cut_obvious_background(image: NDArray):
        return fill_grab_cut_obvious_background(image)

    @staticmethod
    def cvl_fill_grab_cut_obvious_foreground(image: NDArray):
        return fill_grab_cut_obvious_foreground(image)

    @staticmethod
    def cvl_fill_grab_cut_possible_background(image: NDArray):
        return fill_grab_cut_possible_background(image)

    @staticmethod
    def cvl_fill_grab_cut_possible_foreground(image: NDArray):
        return fill_grab_cut_possible_foreground(image)

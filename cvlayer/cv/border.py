# -*- coding: utf-8 -*-

from typing import Optional, Sequence

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.border import (
    DEFAULT_BORDER_TYPE,
    BorderType,
    normalize_border_type,
)
from cvlayer.cv.types.color import ColorLike, normalize_color


def copy_make_border(
    src: NDArray,
    top=1,
    bottom=1,
    left=1,
    right=1,
    border_type=DEFAULT_BORDER_TYPE,
    value: Optional[ColorLike] = None,
    isolated=False,
) -> NDArray:
    _border_type = normalize_border_type(border_type)
    if isolated:
        # When the source image is a part (ROI) of a bigger image,
        # the function will try to use the pixels outside the ROI to form a border.
        # To disable this feature and always do extrapolation, as if src was not a ROI,
        # use `borderType | BORDER_ISOLATED`.
        _border_type |= cv2.BORDER_ISOLATED
    _value = normalize_color(value) if value is not None else None
    return cv2.copyMakeBorder(
        src,
        top,
        bottom,
        left,
        right,
        _border_type,
        None,
        _value,  # type: ignore[arg-type]
    )


def copy_make_border_constant(
    src: NDArray,
    top=1,
    bottom=1,
    left=1,
    right=1,
    value: ColorLike = 0,
    isolated=False,
):
    return copy_make_border(
        src, top, bottom, left, right, BorderType.CONSTANT, value, isolated
    )


def copy_make_border_replicate(
    src: NDArray, top=1, bottom=1, left=1, right=1, isolated=False
):
    return copy_make_border(
        src, top, bottom, left, right, BorderType.REPLICATE, None, isolated
    )


def copy_make_border_reflect(
    src: NDArray, top=1, bottom=1, left=1, right=1, isolated=False
):
    return copy_make_border(
        src, top, bottom, left, right, BorderType.REFLECT, None, isolated
    )


def copy_make_border_wrap(
    src: NDArray, top=1, bottom=1, left=1, right=1, isolated=False
):
    return copy_make_border(
        src, top, bottom, left, right, BorderType.WRAP, None, isolated
    )


def copy_make_border_reflect101(
    src: NDArray, top=1, bottom=1, left=1, right=1, isolated=False
):
    return copy_make_border(
        src, top, bottom, left, right, BorderType.REFLECT101, None, isolated
    )


def copy_make_border_transparent(
    src: NDArray, top=1, bottom=1, left=1, right=1, isolated=False
):
    return copy_make_border(
        src, top, bottom, left, right, BorderType.TRANSPARENT, None, isolated
    )


def copy_make_border_default(
    src: NDArray, top=1, bottom=1, left=1, right=1, isolated=False
):
    return copy_make_border(
        src, top, bottom, left, right, BorderType.DEFAULT, None, isolated
    )


def copy_make_border_isolated(
    src: NDArray, top=1, bottom=1, left=1, right=1, isolated=False
):
    return copy_make_border(
        src, top, bottom, left, right, BorderType.ISOLATED, None, isolated
    )


class CvlBorder:
    @staticmethod
    def cvl_copy_make_border(
        src: NDArray,
        top=1,
        bottom=1,
        left=1,
        right=1,
        border_type=DEFAULT_BORDER_TYPE,
        value: Optional[Sequence[float]] = None,
        isolated=False,
    ):
        return copy_make_border(
            src, top, bottom, left, right, border_type, value, isolated
        )

    @staticmethod
    def cvl_copy_make_border_constant(
        src: NDArray,
        top=1,
        bottom=1,
        left=1,
        right=1,
        value: ColorLike = 0,
        isolated=False,
    ):
        return copy_make_border_constant(src, top, bottom, left, right, value, isolated)

    @staticmethod
    def cvl_copy_make_border_replicate(
        src: NDArray,
        top=1,
        bottom=1,
        left=1,
        right=1,
        isolated=False,
    ):
        return copy_make_border_replicate(src, top, bottom, left, right, isolated)

    @staticmethod
    def cvl_copy_make_border_reflect(
        src: NDArray,
        top=1,
        bottom=1,
        left=1,
        right=1,
        isolated=False,
    ):
        return copy_make_border_reflect(src, top, bottom, left, right, isolated)

    @staticmethod
    def cvl_copy_make_border_wrap(
        src: NDArray,
        top=1,
        bottom=1,
        left=1,
        right=1,
        isolated=False,
    ):
        return copy_make_border_wrap(src, top, bottom, left, right, isolated)

    @staticmethod
    def cvl_copy_make_border_reflect101(
        src: NDArray,
        top=1,
        bottom=1,
        left=1,
        right=1,
        isolated=False,
    ):
        return copy_make_border_reflect101(src, top, bottom, left, right, isolated)

    @staticmethod
    def cvl_copy_make_border_transparent(
        src: NDArray,
        top=1,
        bottom=1,
        left=1,
        right=1,
        isolated=False,
    ):
        return copy_make_border_transparent(src, top, bottom, left, right, isolated)

    @staticmethod
    def cvl_copy_make_border_default(
        src: NDArray,
        top=1,
        bottom=1,
        left=1,
        right=1,
        isolated=False,
    ):
        return copy_make_border_default(src, top, bottom, left, right, isolated)

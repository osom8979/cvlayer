# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final, Optional, Sequence

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.border import BorderType
from cvlayer.typing import PointI, SizeI

DEFAULT_KSIZE: Final[SizeI] = (3, 3)
DEFAULT_ANCHOR: Final[PointI] = (-1, -1)
DEFAULT_ITERATIONS: Final[int] = 1
DEFAULT_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    DEFAULT_KSIZE,
    DEFAULT_ANCHOR,
)


@unique
class MorphShape(Enum):
    RECT = cv2.MORPH_RECT
    CROSS = cv2.MORPH_CROSS
    ELLIPSE = cv2.MORPH_ELLIPSE


@unique
class MorphOperator(Enum):
    ERODE = cv2.MORPH_ERODE
    DILATE = cv2.MORPH_DILATE
    OPEN = cv2.MORPH_OPEN
    CLOSE = cv2.MORPH_CLOSE
    GRADIENT = cv2.MORPH_GRADIENT
    TOPHAT = cv2.MORPH_TOPHAT
    BLACKHAT = cv2.MORPH_BLACKHAT
    HITMISS = cv2.MORPH_HITMISS


def get_structuring_element(
    shape=MorphShape.RECT,
    ksize=DEFAULT_KSIZE,
    anchor=DEFAULT_ANCHOR,
) -> NDArray:
    return cv2.getStructuringElement(shape.value, ksize, anchor)


def get_morph_rect(ksize=DEFAULT_KSIZE, anchor=DEFAULT_ANCHOR) -> NDArray:
    return get_structuring_element(MorphShape.RECT, ksize, anchor)


def get_morph_cross(ksize=DEFAULT_KSIZE, anchor=DEFAULT_ANCHOR) -> NDArray:
    return get_structuring_element(MorphShape.CROSS, ksize, anchor)


def get_morph_ellipse(ksize=DEFAULT_KSIZE, anchor=DEFAULT_ANCHOR) -> NDArray:
    return get_structuring_element(MorphShape.ELLIPSE, ksize, anchor)


def erode(
    src: NDArray,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
    border_type=BorderType.CONSTANT,
    border_value: Optional[Sequence[float]] = None,
) -> NDArray:
    btv = border_type.value
    if border_value:
        return cv2.erode(src, kernel, None, anchor, iterations, btv, border_value)
    else:
        return cv2.erode(src, kernel, None, anchor, iterations, btv)


def dilate(
    src: NDArray,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
    border_type=BorderType.CONSTANT,
    border_value: Optional[Sequence[float]] = None,
) -> NDArray:
    btv = border_type.value
    if border_value:
        return cv2.dilate(src, kernel, None, anchor, iterations, btv, border_value)
    else:
        return cv2.dilate(src, kernel, None, anchor, iterations, btv)


def morphology_ex(
    src: NDArray,
    op: MorphOperator,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
    border_type=BorderType.CONSTANT,
    border_value: Optional[Sequence[float]] = None,
) -> NDArray:
    opv = op.value
    it = iterations
    btv = border_type.value
    if border_value:
        return cv2.morphologyEx(src, opv, kernel, None, anchor, it, btv, border_value)
    else:
        return cv2.morphologyEx(src, opv, kernel, None, anchor, it, btv)


def morphology_ex_erode(
    src: NDArray,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
    border_type=BorderType.CONSTANT,
    border_value: Optional[Sequence[float]] = None,
):
    return morphology_ex(
        src,
        MorphOperator.ERODE,
        kernel,
        anchor,
        iterations,
        border_type,
        border_value,
    )


def morphology_ex_dilate(
    src: NDArray,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
    border_type=BorderType.CONSTANT,
    border_value: Optional[Sequence[float]] = None,
):
    return morphology_ex(
        src,
        MorphOperator.DILATE,
        kernel,
        anchor,
        iterations,
        border_type,
        border_value,
    )


def morphology_ex_open(
    src: NDArray,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
    border_type=BorderType.CONSTANT,
    border_value: Optional[Sequence[float]] = None,
):
    return morphology_ex(
        src,
        MorphOperator.OPEN,
        kernel,
        anchor,
        iterations,
        border_type,
        border_value,
    )


def morphology_ex_close(
    src: NDArray,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
    border_type=BorderType.CONSTANT,
    border_value: Optional[Sequence[float]] = None,
):
    return morphology_ex(
        src,
        MorphOperator.CLOSE,
        kernel,
        anchor,
        iterations,
        border_type,
        border_value,
    )


def morphology_ex_gradient(
    src: NDArray,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
    border_type=BorderType.CONSTANT,
    border_value: Optional[Sequence[float]] = None,
):
    return morphology_ex(
        src,
        MorphOperator.GRADIENT,
        kernel,
        anchor,
        iterations,
        border_type,
        border_value,
    )


def morphology_ex_tophat(
    src: NDArray,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
    border_type=BorderType.CONSTANT,
    border_value: Optional[Sequence[float]] = None,
):
    return morphology_ex(
        src,
        MorphOperator.TOPHAT,
        kernel,
        anchor,
        iterations,
        border_type,
        border_value,
    )


def morphology_ex_blackhat(
    src: NDArray,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
    border_type=BorderType.CONSTANT,
    border_value: Optional[Sequence[float]] = None,
):
    return morphology_ex(
        src,
        MorphOperator.BLACKHAT,
        kernel,
        anchor,
        iterations,
        border_type,
        border_value,
    )


def morphology_ex_hitmiss(
    src: NDArray,
    kernel=DEFAULT_KERNEL,
    anchor=DEFAULT_ANCHOR,
    iterations=DEFAULT_ITERATIONS,
    border_type=BorderType.CONSTANT,
    border_value: Optional[Sequence[float]] = None,
):
    return morphology_ex(
        src,
        MorphOperator.HITMISS,
        kernel,
        anchor,
        iterations,
        border_type,
        border_value,
    )


class CvlMorphology:
    @staticmethod
    def cvl_get_structuring_element(
        shape=MorphShape.RECT,
        ksize=DEFAULT_KSIZE,
        anchor=DEFAULT_ANCHOR,
    ):
        return get_structuring_element(shape, ksize, anchor)

    @staticmethod
    def cvl_get_morph_rect(ksize=DEFAULT_KSIZE, anchor=DEFAULT_ANCHOR):
        return get_morph_rect(ksize, anchor)

    @staticmethod
    def cvl_get_morph_cross(ksize=DEFAULT_KSIZE, anchor=DEFAULT_ANCHOR):
        return get_morph_cross(ksize, anchor)

    @staticmethod
    def cvl_get_morph_ellipse(ksize=DEFAULT_KSIZE, anchor=DEFAULT_ANCHOR):
        return get_morph_ellipse(ksize, anchor)

    @staticmethod
    def cvl_erode(
        src: NDArray,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
        border_type=BorderType.CONSTANT,
        border_value: Optional[Sequence[float]] = None,
    ):
        return erode(src, kernel, anchor, iterations, border_type, border_value)

    @staticmethod
    def cvl_dilate(
        src: NDArray,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
        border_type=BorderType.CONSTANT,
        border_value: Optional[Sequence[float]] = None,
    ):
        return dilate(src, kernel, anchor, iterations, border_type, border_value)

    @staticmethod
    def cvl_morphology_ex(
        src: NDArray,
        op=MorphOperator.OPEN,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
        border_type=BorderType.CONSTANT,
        border_value: Optional[Sequence[float]] = None,
    ):
        return morphology_ex(
            src, op, kernel, anchor, iterations, border_type, border_value
        )

    @staticmethod
    def cvl_morphology_ex_erode(
        src: NDArray,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
        border_type=BorderType.CONSTANT,
        border_value: Optional[Sequence[float]] = None,
    ):
        return morphology_ex_erode(
            src, kernel, anchor, iterations, border_type, border_value
        )

    @staticmethod
    def cvl_morphology_ex_dilate(
        src: NDArray,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
        border_type=BorderType.CONSTANT,
        border_value: Optional[Sequence[float]] = None,
    ):
        return morphology_ex_dilate(
            src, kernel, anchor, iterations, border_type, border_value
        )

    @staticmethod
    def cvl_morphology_ex_open(
        src: NDArray,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
        border_type=BorderType.CONSTANT,
        border_value: Optional[Sequence[float]] = None,
    ):
        return morphology_ex_open(
            src, kernel, anchor, iterations, border_type, border_value
        )

    @staticmethod
    def cvl_morphology_ex_close(
        src: NDArray,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
        border_type=BorderType.CONSTANT,
        border_value: Optional[Sequence[float]] = None,
    ):
        return morphology_ex_close(
            src, kernel, anchor, iterations, border_type, border_value
        )

    @staticmethod
    def cvl_morphology_ex_gradient(
        src: NDArray,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
        border_type=BorderType.CONSTANT,
        border_value: Optional[Sequence[float]] = None,
    ):
        return morphology_ex_gradient(
            src, kernel, anchor, iterations, border_type, border_value
        )

    @staticmethod
    def cvl_morphology_ex_tophat(
        src: NDArray,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
        border_type=BorderType.CONSTANT,
        border_value: Optional[Sequence[float]] = None,
    ):
        return morphology_ex_tophat(
            src, kernel, anchor, iterations, border_type, border_value
        )

    @staticmethod
    def cvl_morphology_ex_blackhat(
        src: NDArray,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
        border_type=BorderType.CONSTANT,
        border_value: Optional[Sequence[float]] = None,
    ):
        return morphology_ex_blackhat(
            src, kernel, anchor, iterations, border_type, border_value
        )

    @staticmethod
    def cvl_morphology_ex_hitmiss(
        src: NDArray,
        kernel=DEFAULT_KERNEL,
        anchor=DEFAULT_ANCHOR,
        iterations=DEFAULT_ITERATIONS,
        border_type=BorderType.CONSTANT,
        border_value: Optional[Sequence[float]] = None,
    ):
        return morphology_ex_hitmiss(
            src, kernel, anchor, iterations, border_type, border_value
        )

# -*- coding: utf-8 -*-

from functools import partial

import cv2
from numpy.typing import NDArray

from cvlayer.cv.types.colormap_type import (
    ColormapType,
    ColormapTypeLike,
    normalize_colormap_type,
)


def apply_colormap(image: NDArray, colormap: ColormapTypeLike) -> NDArray:
    return cv2.applyColorMap(image, normalize_colormap_type(colormap))


apply_colormap_autumn = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_AUTUMN)
apply_colormap_bone = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_BONE)
apply_colormap_jet = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_JET)
apply_colormap_winter = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_WINTER)
apply_colormap_rainbow = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_RAINBOW)
apply_colormap_ocean = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_OCEAN)
apply_colormap_summer = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_SUMMER)
apply_colormap_spring = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_SPRING)
apply_colormap_cool = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_COOL)
apply_colormap_hsv = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_HSV)
apply_colormap_pink = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_PINK)
apply_colormap_hot = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_HOT)
apply_colormap_parula = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_PARULA)
apply_colormap_magma = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_MAGMA)
apply_colormap_inferno = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_INFERNO)
apply_colormap_plasma = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_PLASMA)
apply_colormap_viridis = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_VIRIDIS)
apply_colormap_cividis = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_CIVIDIS)
apply_colormap_twilight = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_TWILIGHT)
apply_colormap_twilight_shifted = partial(
    cv2.applyColorMap, colormap=cv2.COLORMAP_TWILIGHT_SHIFTED
)
apply_colormap_turbo = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_TURBO)
apply_colormap_deepgreen = partial(cv2.applyColorMap, colormap=cv2.COLORMAP_DEEPGREEN)


class CvlColormap:
    @staticmethod
    def cvl_apply_colormap(image: NDArray, colormap: ColormapTypeLike) -> NDArray:
        return apply_colormap(image, colormap)

    @staticmethod
    def cvl_apply_colormap_autumn(image: NDArray):
        return apply_colormap(image, ColormapType.AUTUMN)

    @staticmethod
    def cvl_apply_colormap_bone(image: NDArray):
        return apply_colormap(image, ColormapType.BONE)

    @staticmethod
    def cvl_apply_colormap_jet(image: NDArray):
        return apply_colormap(image, ColormapType.JET)

    @staticmethod
    def cvl_apply_colormap_winter(image: NDArray):
        return apply_colormap(image, ColormapType.WINTER)

    @staticmethod
    def cvl_apply_colormap_rainbow(image: NDArray):
        return apply_colormap(image, ColormapType.RAINBOW)

    @staticmethod
    def cvl_apply_colormap_ocean(image: NDArray):
        return apply_colormap(image, ColormapType.OCEAN)

    @staticmethod
    def cvl_apply_colormap_summer(image: NDArray):
        return apply_colormap(image, ColormapType.SUMMER)

    @staticmethod
    def cvl_apply_colormap_spring(image: NDArray):
        return apply_colormap(image, ColormapType.SPRING)

    @staticmethod
    def cvl_apply_colormap_cool(image: NDArray):
        return apply_colormap(image, ColormapType.COOL)

    @staticmethod
    def cvl_apply_colormap_hsv(image: NDArray):
        return apply_colormap(image, ColormapType.HSV)

    @staticmethod
    def cvl_apply_colormap_pink(image: NDArray):
        return apply_colormap(image, ColormapType.PINK)

    @staticmethod
    def cvl_apply_colormap_hot(image: NDArray):
        return apply_colormap(image, ColormapType.HOT)

    @staticmethod
    def cvl_apply_colormap_parula(image: NDArray):
        return apply_colormap(image, ColormapType.PARULA)

    @staticmethod
    def cvl_apply_colormap_magma(image: NDArray):
        return apply_colormap(image, ColormapType.MAGMA)

    @staticmethod
    def cvl_apply_colormap_inferno(image: NDArray):
        return apply_colormap(image, ColormapType.INFERNO)

    @staticmethod
    def cvl_apply_colormap_plasma(image: NDArray):
        return apply_colormap(image, ColormapType.PLASMA)

    @staticmethod
    def cvl_apply_colormap_viridis(image: NDArray):
        return apply_colormap(image, ColormapType.VIRIDIS)

    @staticmethod
    def cvl_apply_colormap_cividis(image: NDArray):
        return apply_colormap(image, ColormapType.CIVIDIS)

    @staticmethod
    def cvl_apply_colormap_twilight(image: NDArray):
        return apply_colormap(image, ColormapType.TWILIGHT)

    @staticmethod
    def cvl_apply_colormap_twilight_shifted(image: NDArray):
        return apply_colormap(image, ColormapType.TWILIGHT_SHIFTED)

    @staticmethod
    def cvl_apply_colormap_turbo(image: NDArray):
        return apply_colormap(image, ColormapType.TURBO)

    @staticmethod
    def cvl_apply_colormap_deepgreen(image: NDArray):
        return apply_colormap(image, ColormapType.DEEPGREEN)

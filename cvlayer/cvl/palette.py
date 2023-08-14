# -*- coding: utf-8 -*-

from cvlayer.palette import (
    basic_palette,
    css4_palette,
    extended_palette,
    flat_palette,
    tableau_palette,
    xkcd_palette,
)


class CvlPalette:
    @staticmethod
    def palette_basic():
        return basic_palette()

    @staticmethod
    def palette_css4():
        return css4_palette()

    @staticmethod
    def palette_extended():
        return extended_palette()

    @staticmethod
    def palette_flat():
        return flat_palette()

    @staticmethod
    def palette_tableau():
        return tableau_palette()

    @staticmethod
    def palette_xkcd():
        return xkcd_palette()

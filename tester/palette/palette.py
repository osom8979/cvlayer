# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.palette import (
    basic_palette,
    css4_palette,
    extended_palette,
    flat_palette,
    tableau_palette,
    xkcd_palette,
)
from cvlayer.palette.basic import YELLOW
from cvlayer.palette.css4 import DARKGOLDENROD
from cvlayer.palette.extended import DARKSALMON
from cvlayer.palette.flat import PURPLE2
from cvlayer.palette.tableau import BROWN
from cvlayer.palette.xkcd import BURNT_ORANGE


class PaletteTestCase(TestCase):
    def test_basic(self):
        self.assertTupleEqual(YELLOW, basic_palette()["YELLOW"])

    def test_css4(self):
        self.assertTupleEqual(DARKGOLDENROD, css4_palette()["DARKGOLDENROD"])

    def test_extended(self):
        self.assertTupleEqual(DARKSALMON, extended_palette()["DARKSALMON"])

    def test_flat(self):
        self.assertTupleEqual(PURPLE2, flat_palette()["PURPLE2"])

    def test_tableau(self):
        self.assertTupleEqual(BROWN, tableau_palette()["BROWN"])

    def test_xkcd(self):
        self.assertTupleEqual(BURNT_ORANGE, xkcd_palette()["BURNT_ORANGE"])


if __name__ == "__main__":
    main()

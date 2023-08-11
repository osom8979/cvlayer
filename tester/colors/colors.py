# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.colors import (
    basic_colors,
    css4_colors,
    extended_colors,
    flat_colors,
    tableau_colors,
    xkcd_colors,
)
from cvlayer.colors.basic import YELLOW
from cvlayer.colors.css4 import DARKGOLDENROD
from cvlayer.colors.extended import DARKSALMON
from cvlayer.colors.flat import PURPLE2
from cvlayer.colors.tableau import BROWN
from cvlayer.colors.xkcd import BURNT_ORANGE


class ColorsTestCase(TestCase):
    def test_basic(self):
        self.assertTupleEqual(YELLOW, basic_colors()["YELLOW"])

    def test_css4(self):
        self.assertTupleEqual(DARKGOLDENROD, css4_colors()["DARKGOLDENROD"])

    def test_extended(self):
        self.assertTupleEqual(DARKSALMON, extended_colors()["DARKSALMON"])

    def test_flat(self):
        self.assertTupleEqual(PURPLE2, flat_colors()["PURPLE2"])

    def test_tableau(self):
        self.assertTupleEqual(BROWN, tableau_colors()["BROWN"])

    def test_xkcd(self):
        self.assertTupleEqual(BURNT_ORANGE, xkcd_colors()["BURNT_ORANGE"])


if __name__ == "__main__":
    main()

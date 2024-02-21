# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.math.angle import (
    normalize_degrees_360,
    normalize_signed_degrees_180,
    normalize_signed_degrees_360,
)


class NormalizeDegreesTestCase(TestCase):
    def test_normalize_degrees_360(self):
        self.assertEqual(0, normalize_degrees_360(0))
        self.assertEqual(1, normalize_degrees_360(1))
        self.assertEqual(359, normalize_degrees_360(-1))
        self.assertEqual(180, normalize_degrees_360(180))
        self.assertEqual(180, normalize_degrees_360(-180))
        self.assertEqual(0, normalize_degrees_360(360))
        self.assertEqual(0, normalize_degrees_360(-360))
        self.assertEqual(180, normalize_degrees_360(540))
        self.assertEqual(180, normalize_degrees_360(-540))
        self.assertEqual(0, normalize_degrees_360(720))
        self.assertEqual(0, normalize_degrees_360(-720))
        self.assertEqual(1, normalize_degrees_360(721))
        self.assertEqual(359, normalize_degrees_360(-721))

    def test_normalize_signed_degrees_360(self):
        self.assertEqual(0, normalize_signed_degrees_360(0))
        self.assertEqual(1, normalize_signed_degrees_360(1))
        self.assertEqual(-1, normalize_signed_degrees_360(-1))
        self.assertEqual(180, normalize_signed_degrees_360(180))
        self.assertEqual(-180, normalize_signed_degrees_360(-180))
        self.assertEqual(0, normalize_signed_degrees_360(360))
        self.assertEqual(0, normalize_signed_degrees_360(-360))
        self.assertEqual(180, normalize_signed_degrees_360(540))
        self.assertEqual(-180, normalize_signed_degrees_360(-540))
        self.assertEqual(0, normalize_signed_degrees_360(720))
        self.assertEqual(0, normalize_signed_degrees_360(-720))
        self.assertEqual(1, normalize_signed_degrees_360(721))
        self.assertEqual(-1, normalize_signed_degrees_360(-721))

    def test_normalize_signed_degrees_180(self):
        self.assertEqual(0, normalize_signed_degrees_180(0))
        self.assertEqual(1, normalize_signed_degrees_180(1))
        self.assertEqual(-1, normalize_signed_degrees_180(-1))
        self.assertEqual(90, normalize_signed_degrees_180(90))
        self.assertEqual(-90, normalize_signed_degrees_180(-90))
        self.assertEqual(180, normalize_signed_degrees_180(180))
        self.assertEqual(180, normalize_signed_degrees_180(-180))
        self.assertEqual(-179, normalize_signed_degrees_180(181))
        self.assertEqual(179, normalize_signed_degrees_180(-181))
        self.assertEqual(-90, normalize_signed_degrees_180(270))
        self.assertEqual(90, normalize_signed_degrees_180(-270))

        self.assertEqual(0, normalize_signed_degrees_180(360))
        self.assertEqual(0, normalize_signed_degrees_180(-360))
        self.assertEqual(1, normalize_signed_degrees_180(361))
        self.assertEqual(-1, normalize_signed_degrees_180(-361))
        self.assertEqual(90, normalize_signed_degrees_180(450))
        self.assertEqual(-90, normalize_signed_degrees_180(-450))
        self.assertEqual(180, normalize_signed_degrees_180(540))
        self.assertEqual(180, normalize_signed_degrees_180(-540))
        self.assertEqual(-179, normalize_signed_degrees_180(541))
        self.assertEqual(179, normalize_signed_degrees_180(-541))
        self.assertEqual(-90, normalize_signed_degrees_180(630))
        self.assertEqual(90, normalize_signed_degrees_180(-630))

        self.assertEqual(0, normalize_signed_degrees_180(720))
        self.assertEqual(0, normalize_signed_degrees_180(-720))
        self.assertEqual(1, normalize_signed_degrees_180(721))
        self.assertEqual(-1, normalize_signed_degrees_180(-721))


if __name__ == "__main__":
    main()

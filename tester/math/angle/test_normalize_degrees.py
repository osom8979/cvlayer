# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.math.angle import normalize_degrees_360, normalize_signed_degrees_360


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


if __name__ == "__main__":
    main()

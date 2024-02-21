# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.rotate.rotate_tracer import normalize_signed_degrees_180


class RotateTracerTestCase(TestCase):
    def test_normalize_degrees_180(self):
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


if __name__ == "__main__":
    main()

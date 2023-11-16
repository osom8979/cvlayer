# -*- coding: utf-8 -*-

from math import degrees
from unittest import TestCase, main

from cvlayer.math.angle import degrees_point3
from cvlayer.math.constant import DOUBLE_PI, PI


class DegreesPoint3TestCase(TestCase):
    def setUp(self):
        self.center = (0, 0)
        self.bottom = (0, -1)
        self.top = (0, 1)
        self.right = (1, 0)
        self.left = (-1, 0)

    def test_degrees(self):
        self.assertEqual(0, degrees(0))
        self.assertEqual(180, degrees(PI))
        self.assertEqual(360, degrees(DOUBLE_PI))

    def test_degrees_point3_pivot_right(self):
        #################
        #       90      #
        #       |       #
        # 180 --c--  p  #
        #       |       #
        #      270      #
        #################
        p2r = degrees_point3(self.right, self.center, self.right)
        p2t = degrees_point3(self.right, self.center, self.top)
        p2l = degrees_point3(self.right, self.center, self.left)
        p2b = degrees_point3(self.right, self.center, self.bottom)
        self.assertEqual(0, p2r)
        self.assertEqual(90, p2t)
        self.assertEqual(180, p2l)
        self.assertEqual(270, p2b)

    def test_degrees_point3_pivot_top(self):
        #################
        #       p       #
        #       |       #
        #  90 --c-- 270 #
        #       |       #
        #      180      #
        #################
        p2r = degrees_point3(self.top, self.center, self.right)
        p2t = degrees_point3(self.top, self.center, self.top)
        p2l = degrees_point3(self.top, self.center, self.left)
        p2b = degrees_point3(self.top, self.center, self.bottom)
        self.assertEqual(270, p2r)
        self.assertEqual(0, p2t)
        self.assertEqual(90, p2l)
        self.assertEqual(180, p2b)

    def test_degrees_point3_pivot_left(self):
        #################
        #      270      #
        #       |       #
        #  p  --c-- 180 #
        #       |       #
        #       90      #
        #################
        p2r = degrees_point3(self.left, self.center, self.right)
        p2t = degrees_point3(self.left, self.center, self.top)
        p2l = degrees_point3(self.left, self.center, self.left)
        p2b = degrees_point3(self.left, self.center, self.bottom)
        self.assertEqual(180, p2r)
        self.assertEqual(270, p2t)
        self.assertEqual(0, p2l)
        self.assertEqual(90, p2b)

    def test_degrees_point3_pivot_bottom(self):
        #################
        #      180      #
        #       |       #
        # 270 --c--  90 #
        #       |       #
        #       p       #
        #################
        p2r = degrees_point3(self.bottom, self.center, self.right)
        p2t = degrees_point3(self.bottom, self.center, self.top)
        p2l = degrees_point3(self.bottom, self.center, self.left)
        p2b = degrees_point3(self.bottom, self.center, self.bottom)
        self.assertEqual(90, p2r)
        self.assertEqual(180, p2t)
        self.assertEqual(270, p2l)
        self.assertEqual(0, p2b)


if __name__ == "__main__":
    main()

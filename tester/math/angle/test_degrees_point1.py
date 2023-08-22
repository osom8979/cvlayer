# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.math.angle import degrees_point1


class DegreesPoint1TestCase(TestCase):
    def setUp(self):
        self.bottom = (0, -1)
        self.top = (0, 1)
        self.right = (1, 0)
        self.left = (-1, 0)

    def test_degrees_angle_point1(self):
        #################
        #       90      #
        #       |       #
        # 180 --+--  0  #
        #       |       #
        #      270      #
        #################
        self.assertEqual(270, degrees_point1(self.bottom))
        self.assertEqual(180, degrees_point1(self.left))
        self.assertEqual(90, degrees_point1(self.top))
        self.assertEqual(0, degrees_point1(self.right))


if __name__ == "__main__":
    main()

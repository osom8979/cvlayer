# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.rotate.rotate_meter import RotateMeter


class RotateMeterTestCase(TestCase):
    @staticmethod
    def make_moved_center_and_point(cx, cy, px, py, d):
        return (cx + d, cy + d), (px + d, py + d)

    def test_45_first(self):
        #################
        #  y1  +90   x  #
        #       |       #
        # 180 --+--  0  #
        #       |       #
        #  y2  270   y3 #
        #################
        rm = RotateMeter()
        rm.set_first(*self.make_moved_center_and_point(0, 0, 1, 1, 4.5))
        rm.set_second(*self.make_moved_center_and_point(0, 0, -1, 1, -9.0))
        self.assertEqual(90, rm.angle)

        rm.clear()
        rm.set_first(*self.make_moved_center_and_point(0, 0, 1, 1, 3.5))
        rm.set_second(*self.make_moved_center_and_point(0, 0, -1, -1, -8.3))
        self.assertEqual(180, rm.angle)

        rm.clear()
        rm.set_first(*self.make_moved_center_and_point(0, 0, 1, 1, -7.2))
        rm.set_second(*self.make_moved_center_and_point(0, 0, 1, -1, 6.6))
        self.assertEqual(270, rm.angle)

    def test_135_first(self):
        #################
        #  x   +90   y3 #
        #       |       #
        # 180 --+--  0  #
        #       |       #
        #  y1  270   y2 #
        #################
        rm = RotateMeter()
        rm.set_first(*self.make_moved_center_and_point(0, 0, -1, 1, -9.0))
        rm.set_second(*self.make_moved_center_and_point(0, 0, -1, -1, 10.5))
        self.assertEqual(90, rm.angle)

        rm.clear()
        rm.set_first(*self.make_moved_center_and_point(0, 0, -1, 1, -19.1))
        rm.set_second(*self.make_moved_center_and_point(0, 0, 1, -1, 11.1))
        self.assertEqual(180, rm.angle)

        rm.clear()
        rm.set_first(*self.make_moved_center_and_point(0, 0, -1, 1, -10.2))
        rm.set_second(*self.make_moved_center_and_point(0, 0, 1, 1, -10))
        self.assertEqual(270, rm.angle)

    def test_225_first(self):
        #################
        #  y3  +90   y2 #
        #       |       #
        # 180 --+--  0  #
        #       |       #
        #  x   270   y1 #
        #################
        rm = RotateMeter()
        rm.set_first(*self.make_moved_center_and_point(0, 0, -1, -1, -9.0))
        rm.set_second(*self.make_moved_center_and_point(0, 0, 1, -1, 8.99))
        self.assertEqual(90, rm.angle)

        rm.clear()
        rm.set_first(*self.make_moved_center_and_point(0, 0, -1, -1, -38.0))
        rm.set_second(*self.make_moved_center_and_point(0, 0, 1, 1, 12.2))
        self.assertEqual(180, rm.angle)

        rm.clear()
        rm.set_first(*self.make_moved_center_and_point(0, 0, -1, -1, 23.0))
        rm.set_second(*self.make_moved_center_and_point(0, 0, -1, 1, -84.1))
        self.assertEqual(270, rm.angle)

    def test_315_first(self):
        #################
        #  y2  +90   y1 #
        #       |       #
        # 180 --+--  0  #
        #       |       #
        #  y3  270   x  #
        #################
        rm = RotateMeter()
        rm.set_first(*self.make_moved_center_and_point(0, 0, 1, -1, 0.2))
        rm.set_second(*self.make_moved_center_and_point(0, 0, 1, 1, 99))
        self.assertEqual(90, rm.angle)

        rm.clear()
        rm.set_first(*self.make_moved_center_and_point(0, 0, 1, -1, -0.1))
        rm.set_second(*self.make_moved_center_and_point(0, 0, -1, 1, 1.3))
        self.assertEqual(180, rm.angle)

        rm.clear()
        rm.set_first(*self.make_moved_center_and_point(0, 0, 1, -1, -75.8))
        rm.set_second(*self.make_moved_center_and_point(0, 0, -1, -1, -38.3))
        self.assertEqual(270, rm.angle)


if __name__ == "__main__":
    main()

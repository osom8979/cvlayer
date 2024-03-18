# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.rotate.rotate_tracer import RotateTracer


class RotateTracerTestCase(TestCase):
    @staticmethod
    def make_center_point(cx, cy, px, py, d):
        return (cx + d, cy + d), (px + d, py + d)

    def test_rotate(self):
        #################
        #  p2  +90   p1 #
        #       |       #
        # 180 --+--  0  #
        #       |       #
        #  p3  270   p4 #
        #################
        ra = RotateTracer()
        self.assertEqual(0, ra.push(*self.make_center_point(0, 0, 1, 1, 4.5)))
        self.assertEqual(90, ra.push(*self.make_center_point(0, 0, -1, 1, -9)))
        self.assertEqual(180, ra.push(*self.make_center_point(0, 0, -1, -1, 8)))
        self.assertEqual(270, ra.push(*self.make_center_point(0, 0, 1, -1, -6)))
        self.assertEqual(360, ra.push(*self.make_center_point(0, 0, 1, 1, 4)))
        self.assertEqual(450, ra.push(*self.make_center_point(0, 0, -1, 1, -8)))

        self.assertEqual(360, ra.push(*self.make_center_point(0, 0, 1, 1, 4)))
        self.assertEqual(270, ra.push(*self.make_center_point(0, 0, 1, -1, -6)))
        self.assertEqual(180, ra.push(*self.make_center_point(0, 0, -1, -1, 8)))
        self.assertEqual(90, ra.push(*self.make_center_point(0, 0, -1, 1, -9)))
        self.assertEqual(0, ra.push(*self.make_center_point(0, 0, 1, 1, 4.5)))

        self.assertEqual(-90, ra.push(*self.make_center_point(0, 0, 1, -1, -6)))
        self.assertEqual(-180, ra.push(*self.make_center_point(0, 0, -1, -1, 8)))
        self.assertEqual(-270, ra.push(*self.make_center_point(0, 0, -1, 1, -9)))
        self.assertEqual(-360, ra.push(*self.make_center_point(0, 0, 1, 1, 4.5)))
        self.assertEqual(-450, ra.push(*self.make_center_point(0, 0, 1, -1, -6)))


if __name__ == "__main__":
    main()

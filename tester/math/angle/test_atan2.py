# -*- coding: utf-8 -*-

from math import atan2, pi
from unittest import TestCase, main


class Atan2TestCase(TestCase):
    def test_atan2(self):
        #################
        #      +90      #
        #       |       #
        # 180 --+--  0  #
        #  x    |       #
        #      -90      #
        #################
        half_pi = pi / 2

        self.assertEqual(0, atan2(0, 1))
        self.assertAlmostEqual(half_pi, atan2(1, 0))
        self.assertAlmostEqual(pi, atan2(0, -1))
        self.assertAlmostEqual(-half_pi, atan2(-1, 0))

        x = atan2(-0.00000001, -1)
        self.assertGreater(pi, -x)


if __name__ == "__main__":
    main()

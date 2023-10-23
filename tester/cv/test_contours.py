# -*- coding: utf-8 -*-

from unittest import TestCase, main

import cv2
from numpy import int32, ndarray

from cvlayer.cv.contours import FindContoursMethod, FindContoursMode, find_contours
from cvlayer.cv.cvt_color import cvt_color_BGR2GRAY, cvt_color_GRAY2BGR
from cvlayer.cv.drawable.rectangle import draw_rectangle
from cvlayer.cv.image_io import image_write
from cvlayer.cv.image_make import make_image_empty
from cvlayer.cv.types.thickness import FILLED
from cvlayer.palette.basic import BLACK, WHITE


class ContoursTestCase(TestCase):
    def setUp(self):
        image = make_image_empty(700, 700)
        draw_rectangle(image, (100, 100, 600, 600), WHITE, FILLED)
        draw_rectangle(image, (200, 200, 500, 500), BLACK, FILLED)
        draw_rectangle(image, (300, 300, 400, 400), WHITE, FILLED)
        self.image = cvt_color_BGR2GRAY(image)

    def test_find_contours_tree_simple(self):
        mode = FindContoursMode.TREE
        method = FindContoursMethod.SIMPLE
        result = find_contours(self.image, mode, method)
        contours = result.contours
        hierarchy = result.hierarchy
        self.assertEqual(3, len(contours))

        c0 = contours[0]
        c1 = contours[1]
        c2 = contours[2]
        self.assertIsInstance(c0, ndarray)
        self.assertIsInstance(c1, ndarray)
        self.assertIsInstance(c2, ndarray)
        self.assertTupleEqual((4, 1, 2), c0.shape)
        c0_expect = [[100, 100], [100, 600], [600, 600], [600, 100]]
        self.assertListEqual(c0_expect, c0[:, 0, :].tolist())
        # self.assertTupleEqual((8, 1, 2), c1.shape)
        # self.assertTupleEqual((4, 1, 2), c2.shape)
        self.assertEqual(int32, c0.dtype)
        self.assertEqual(int32, c1.dtype)
        self.assertEqual(int32, c2.dtype)
        self.image = cvt_color_GRAY2BGR(self.image)
        cv2.drawContours(self.image, contours, 1, (0, 0, 255), 2)
        image_write("aaa.png", self.image)

        self.assertIsInstance(hierarchy, ndarray)
        self.assertEqual((1, 3, 4), hierarchy.shape)
        self.assertEqual(int32, hierarchy.dtype)


if __name__ == "__main__":
    main()

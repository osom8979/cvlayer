# -*- coding: utf-8 -*-

from unittest import TestCase, main

from numpy import int32, ndarray

from cvlayer.cv.contour.find import find_contours
from cvlayer.cv.cvt_color import cvt_color_BGR2GRAY, cvt_color_GRAY2BGR
from cvlayer.cv.drawable.contours import draw_contours
from cvlayer.cv.drawable.rectangle import draw_rectangle
from cvlayer.cv.image_io import image_write
from cvlayer.cv.image_make import make_image_empty
from cvlayer.cv.types.chain_approx import ChainApproximation
from cvlayer.cv.types.retrieval import Retrieval
from cvlayer.cv.types.thickness import FILLED
from cvlayer.palette.basic import BLACK, RED, WHITE


class FindTestCase(TestCase):
    def setUp(self):
        image = make_image_empty(700, 700)
        draw_rectangle(image, (100, 100, 600, 600), WHITE, FILLED)
        draw_rectangle(image, (200, 200, 500, 500), BLACK, FILLED)
        draw_rectangle(image, (300, 300, 400, 400), WHITE, FILLED)
        self.image = cvt_color_BGR2GRAY(image)
        self.write_file = False

    def test_find_contours_tree_simple(self):
        mode = Retrieval.TREE
        method = ChainApproximation.SIMPLE
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
        self.assertTupleEqual((8, 1, 2), c0.shape)
        self.assertTupleEqual((8, 1, 2), c1.shape)
        self.assertTupleEqual((8, 1, 2), c2.shape)
        self.assertEqual(int32, c0.dtype)
        self.assertEqual(int32, c1.dtype)
        self.assertEqual(int32, c2.dtype)

        self.assertIsInstance(hierarchy, ndarray)
        self.assertEqual((1, 3, 4), hierarchy.shape)
        self.assertEqual(int32, hierarchy.dtype)

        if self.write_file:
            image = cvt_color_GRAY2BGR(self.image)
            draw_contours(image, contours, color=RED, hierarchy=hierarchy)
            image_write("test_find_contours_tree_simple.png", image)


if __name__ == "__main__":
    main()

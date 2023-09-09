# -*- coding: utf-8 -*-

from unittest import TestCase, main

from numpy import all as np_all

from cvlayer.cv.image_crop import image_crop, image_crop_adjusted, image_crop_extended
from cvlayer.cv.image_make import make_image_filled
from cvlayer.palette.basic import BLACK, RED


class ImageCropTestCase(TestCase):
    def test_image_crop(self):
        src = make_image_filled(10, 10, BLACK)

        img1 = image_crop(src, (5, 5, 9, 9))
        self.assertTupleEqual((4, 4, 3), img1.shape)

        img2 = image_crop(src, (5, 5, 10, 10))
        self.assertTupleEqual((5, 5, 3), img2.shape)

        with self.assertRaises(IndexError):
            image_crop(src, (5, 5, 11, 11))
        with self.assertRaises(IndexError):
            image_crop(src, (9, 9, 12, 12))
        with self.assertRaises(IndexError):
            image_crop(src, (-1, -1, 5, 5))

    def test_image_crop_adjusted(self):
        src = make_image_filled(10, 10, BLACK)

        img1 = image_crop_adjusted(src, (5, 5, 9, 9))
        self.assertTupleEqual((4, 4, 3), img1.shape)

        img2 = image_crop_adjusted(src, (5, 5, 10, 10))
        self.assertTupleEqual((5, 5, 3), img2.shape)

        img3 = image_crop_adjusted(src, (5, 5, 11, 11))
        self.assertTupleEqual((5, 5, 3), img3.shape)

        img4 = image_crop_adjusted(src, (9, 9, 12, 12))
        self.assertTupleEqual((1, 1, 3), img4.shape)

        img5 = image_crop_adjusted(src, (-1, -1, 5, 5))
        self.assertTupleEqual((5, 5, 3), img5.shape)

    def test_image_crop_extended_size(self):
        src = make_image_filled(10, 10, BLACK)

        img1 = image_crop_extended(src, (5, 5, 9, 9))
        self.assertTupleEqual((4, 4, 3), img1.shape)

        img2 = image_crop_extended(src, (5, 5, 10, 10))
        self.assertTupleEqual((5, 5, 3), img2.shape)

        img3 = image_crop_extended(src, (5, 5, 11, 11))
        self.assertTupleEqual((6, 6, 3), img3.shape)

        img4 = image_crop_extended(src, (9, 9, 12, 12))
        self.assertTupleEqual((3, 3, 3), img4.shape)

        img5 = image_crop_extended(src, (-1, -1, 5, 5))
        self.assertTupleEqual((6, 6, 3), img5.shape)

    def test_image_crop_extended_data(self):
        src = make_image_filled(1, 1, RED)

        img1 = image_crop_extended(src, (0, 0, 1, 1), BLACK)
        self.assertTupleEqual((1, 1, 3), img1.shape)
        self.assertTrue(np_all(img1 == RED))

        img2 = image_crop_extended(src, (0, 0, 2, 2), BLACK)
        self.assertTupleEqual((2, 2, 3), img2.shape)
        self.assertTrue(np_all(img2[0, 0] == RED))
        self.assertTrue(np_all(img2[0, 1] == BLACK))
        self.assertTrue(np_all(img2[1, 0] == BLACK))
        self.assertTrue(np_all(img2[1, 1] == BLACK))

        img3 = image_crop_extended(src, (-1, -1, 1, 1), BLACK)
        self.assertTupleEqual((2, 2, 3), img3.shape)
        self.assertTrue(np_all(img3[0, 0] == BLACK))
        self.assertTrue(np_all(img3[0, 1] == BLACK))
        self.assertTrue(np_all(img3[1, 0] == BLACK))
        self.assertTrue(np_all(img3[1, 1] == RED))


if __name__ == "__main__":
    main()

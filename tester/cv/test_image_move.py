# -*- coding: utf-8 -*-

from unittest import TestCase, main

from numpy import all as np_all

from cvlayer.cv.drawable.rectangle import draw_rectangle_coord
from cvlayer.cv.image_make import make_image_filled
from cvlayer.cv.image_move import move
from cvlayer.cv.types.line_type import LINE_4
from cvlayer.cv.types.thickness import FILLED
from cvlayer.palette.basic import BLACK, RED


class ImageMoveTestCase(TestCase):
    def test_image_move(self):
        src = make_image_filled(10, 10, BLACK)
        draw_rectangle_coord(src, 1, 1, 2, 2, RED, FILLED, LINE_4)

        actually = move(src, 6, 4)

        expected = make_image_filled(10, 10, BLACK)
        draw_rectangle_coord(expected, 7, 5, 8, 6, RED, FILLED, LINE_4)
        self.assertTrue(np_all(actually == expected))


if __name__ == "__main__":
    main()

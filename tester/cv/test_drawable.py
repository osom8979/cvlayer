# -*- coding: utf-8 -*-

from os import getcwd, path
from shutil import copy
from tempfile import TemporaryDirectory
from typing import Final
from unittest import TestCase, main

from cvlayer.cv.drawable import draw_multiline_text_box
from cvlayer.cv.image_io import image_write
from cvlayer.cv.image_make import make_image_filled
from cvlayer.palette.basic import RED

ENABLE_OUTPUT: Final[bool] = False
OUTPUT_DIR: Final[str] = getcwd()


class DrawableTestCase(TestCase):
    def setUp(self):
        self.enable_output = ENABLE_OUTPUT
        self.output_dir = OUTPUT_DIR

    def test_draw_multiline_text_box(self):
        with TemporaryDirectory() as tmpdir:
            self.assertTrue(path.isdir(tmpdir))

            bg = make_image_filled(500, 500, RED)
            x = 0
            y = 0
            text = "DrawableTestCase\n1234567890\n!@#$%^&*()"

            img1 = bg.copy()
            draw_multiline_text_box(img1, text, x, y)
            file1 = path.join(tmpdir, "img1.png")
            self.assertTrue(image_write(file1, img1))
            if self.enable_output:
                copy(file1, self.output_dir)

            img2 = bg.copy()
            draw_multiline_text_box(img2, text, x, y, anchor_x=0.5, anchor_y=0.5)
            file2 = path.join(tmpdir, "img2.png")
            self.assertTrue(image_write(file2, img2))
            if self.enable_output:
                copy(file2, self.output_dir)

            img3 = bg.copy()
            draw_multiline_text_box(img3, text, x, y, anchor_x=1, anchor_y=1)
            file3 = path.join(tmpdir, "img3.png")
            self.assertTrue(image_write(file3, img3))
            if self.enable_output:
                copy(file3, self.output_dir)


if __name__ == "__main__":
    main()

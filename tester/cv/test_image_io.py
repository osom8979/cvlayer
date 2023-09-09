# -*- coding: utf-8 -*-

from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from numpy import all as np_all

from cvlayer.cv.image_io import image_read, image_write
from cvlayer.cv.image_make import make_image_random


class ImageIoTestCase(TestCase):
    def test_io(self):
        with TemporaryDirectory() as tmpdir:
            self.assertTrue(path.isdir(tmpdir))

            img1 = make_image_random(10, 10)
            filename = path.join(tmpdir, "img1.png")
            self.assertTrue(image_write(filename, img1))

            img2 = image_read(filename)
            self.assertTrue(np_all(img2 == img1))


if __name__ == "__main__":
    main()

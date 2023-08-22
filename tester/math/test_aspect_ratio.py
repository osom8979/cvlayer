# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.math.aspect_ratio import aspect_ratio, rescale_aspect_ratio


class AspectRatioTestCase(TestCase):
    def test_aspect_ratio(self):
        self.assertEqual((16, 9), aspect_ratio(1920, 1080))

    def test_rescale_aspect_ratio(self):
        self.assertEqual((16, 9), rescale_aspect_ratio(1920, 1080, 16, None))
        self.assertEqual((16, 9), rescale_aspect_ratio(1920, 1080, None, 9))


if __name__ == "__main__":
    main()

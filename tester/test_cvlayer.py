# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer import CvLayer


class CvLayerTestCase(TestCase, CvLayer):
    def test_cvt_color_code_type(self):
        self.assertIsInstance(self.CvtColorCodeType.GRAY2BGR.value, int)


if __name__ == "__main__":
    main()

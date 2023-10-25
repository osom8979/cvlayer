# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.cv.types.border import DEFAULT_BORDER_TYPE, BorderType, normalize_border


class BorderTestCase(TestCase):
    def test_defaults(self):
        b1 = normalize_border(DEFAULT_BORDER_TYPE)
        b2 = normalize_border(None)
        b3 = normalize_border("default")
        self.assertEqual(BorderType.DEFAULT.value, b1)
        self.assertEqual(BorderType.DEFAULT.value, b2)
        self.assertEqual(BorderType.DEFAULT.value, b3)

    def test_normalize_border_str(self):
        self.assertEqual(BorderType.CONSTANT.value, normalize_border("CONSTANT"))
        self.assertEqual(BorderType.REPLICATE.value, normalize_border("REPLICATE"))
        self.assertEqual(BorderType.REFLECT.value, normalize_border("REFLECT"))
        self.assertEqual(BorderType.WRAP.value, normalize_border("WRAP"))
        self.assertEqual(BorderType.REFLECT101.value, normalize_border("REFLECT101"))
        self.assertEqual(BorderType.TRANSPARENT.value, normalize_border("TRANSPARENT"))
        self.assertEqual(BorderType.DEFAULT.value, normalize_border("DEFAULT"))
        self.assertEqual(BorderType.ISOLATED.value, normalize_border("ISOLATED"))


if __name__ == "__main__":
    main()

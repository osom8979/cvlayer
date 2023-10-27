# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.cv.types.border import (
    BORDER_CONSTANT,
    BORDER_DEFAULT,
    BORDER_ISOLATED,
    BORDER_REFLECT,
    BORDER_REFLECT101,
    BORDER_REPLICATE,
    BORDER_TRANSPARENT,
    BORDER_WRAP,
    DEFAULT_BORDER_TYPE,
    normalize_border_type,
)


class BorderTestCase(TestCase):
    def test_defaults(self):
        b1 = normalize_border_type(DEFAULT_BORDER_TYPE)
        b2 = normalize_border_type(None)
        b3 = normalize_border_type("default")
        self.assertEqual(BORDER_DEFAULT, b1)
        self.assertEqual(BORDER_DEFAULT, b2)
        self.assertEqual(BORDER_DEFAULT, b3)

    def test_normalize_border_str(self):
        self.assertEqual(BORDER_CONSTANT, normalize_border_type("CONSTANT"))
        self.assertEqual(BORDER_REPLICATE, normalize_border_type("REPLICATE"))
        self.assertEqual(BORDER_REFLECT, normalize_border_type("REFLECT"))
        self.assertEqual(BORDER_WRAP, normalize_border_type("WRAP"))
        self.assertEqual(BORDER_REFLECT101, normalize_border_type("REFLECT101"))
        self.assertEqual(BORDER_TRANSPARENT, normalize_border_type("TRANSPARENT"))
        self.assertEqual(BORDER_DEFAULT, normalize_border_type("DEFAULT"))
        self.assertEqual(BORDER_ISOLATED, normalize_border_type("ISOLATED"))


if __name__ == "__main__":
    main()

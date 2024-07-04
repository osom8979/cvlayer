# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.cv.stitching.props import StitcherProps
from cvlayer.inspect.docstring import get_attribute_docstring


class DocstringTestCase(TestCase):
    def test_get_attribute_docstring(self):
        doc0 = get_attribute_docstring(StitcherProps, "blend_strength")
        self.assertEqual("Blend Strength", doc0)

        doc1 = get_attribute_docstring(StitcherProps, "stitcher_mode_index")
        self.assertEqual("Scenario for stitcher operation.", doc1.split("\n")[0])


if __name__ == "__main__":
    main()

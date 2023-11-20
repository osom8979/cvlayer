# -*- coding: utf-8 -*-

from unittest import TestCase, main

from numpy import uint8, zeros

from cvlayer.cv.cvt_color import cvt_color_GRAY2BGR
from cvlayer.cv.drawable.crosshair import draw_crosshair
from cvlayer.cv.drawable.line import draw_line
from cvlayer.cv.drawable.rectangle import draw_rectangle
from cvlayer.cv.image_io import image_write
from cvlayer.cv.intersection import intersection_mask_line
from cvlayer.palette.basic import BLUE, GRAY, LIME, RED, YELLOW


class IntersectionTestCase(TestCase):
    def setUp(self):
        self.x_line = (40, 90), (160, 110)
        self.y_line = (90, 40), (110, 160)
        self.mask = zeros((200, 200), dtype=uint8)
        draw_rectangle(self.mask, (50, 50, 150, 150), color=255, thickness=1, line=4)
        self.write_file = False

    def test_intersection_mask_line_x(self):
        result = intersection_mask_line(self.mask, self.x_line)
        self.assertEqual(result.leftmost[0], 50)
        self.assertEqual(result.rightmost[0], 150)
        self.assertTupleEqual(result.leftmost, result.topmost)
        self.assertTupleEqual(result.rightmost, result.bottommost)

        if self.write_file:
            image = cvt_color_GRAY2BGR(self.mask)
            draw_line(image, self.x_line[0], self.x_line[1], GRAY, thickness=1, line=4)
            draw_crosshair(image, result.leftmost, color=BLUE, line=4)
            draw_crosshair(image, result.rightmost, color=LIME, line=4)
            draw_crosshair(image, result.topmost, color=RED, line=4, circle=False)
            draw_crosshair(image, result.bottommost, color=YELLOW, line=4, circle=False)
            image_write("test_intersection_mask_line_x.png", image)

    def test_intersection_mask_line_y(self):
        result = intersection_mask_line(self.mask, self.y_line)
        self.assertEqual(result.topmost[1], 50)
        self.assertEqual(result.bottommost[1], 150)
        self.assertTupleEqual(result.leftmost, result.topmost)
        self.assertTupleEqual(result.rightmost, result.bottommost)

        if self.write_file:
            image = cvt_color_GRAY2BGR(self.mask)
            draw_line(image, self.y_line[0], self.y_line[1], GRAY, thickness=1, line=4)
            draw_crosshair(image, result.leftmost, color=BLUE, line=4)
            draw_crosshair(image, result.rightmost, color=LIME, line=4)
            draw_crosshair(image, result.topmost, color=RED, line=4, circle=False)
            draw_crosshair(image, result.bottommost, color=YELLOW, line=4, circle=False)
            image_write("test_intersection_mask_line_y.png", image)


if __name__ == "__main__":
    main()

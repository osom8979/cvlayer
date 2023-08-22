# -*- coding: utf-8 -*-

from unittest import TestCase, main

from shapely.geometry import Polygon

from cvlayer.cv.contours_shapely import intersection_polygon


class ContoursShapelyTestCase(TestCase):
    def test_intersection_polygon_single_polygon(self):
        # ==========================
        #           +y             |
        #            |             |
        #            |             |
        #     (-1,1) |             |
        #        *-------*         |
        #        |   |   |         |
        # -x ----|---*---|---*--- +x
        #        |   |   |   |     |
        #        *-------*   |     |
        #            |       |     |
        #            *-------*     |
        #            |     (2,-2)  |
        #           -y             |
        # ==========================
        roi1 = (-1, 1), (1, 1), (1, -1), (-1, -1)
        roi2 = (0, 0), (2, 0), (2, -2), (0, -2)
        polygon1 = Polygon(roi1)
        polygon2 = Polygon(roi2)
        result = intersection_polygon(polygon1, polygon2)
        self.assertEqual(1, len(result))

        expected_bbox = [0, -1, 1, 0]
        result_bbox = list(round(i) for i in result[0].bounds)
        self.assertEqual(expected_bbox, result_bbox)

    def test_intersection_polygon_single_line(self):
        # ==========================
        #           +y             |
        #            |     (2,2)   |
        #            *-------*     |
        #            |       |     |
        #            |       |     |
        #            |       |     |
        # -x --------*-------*--- +x
        #            |       |     |
        #            |       |     |
        #            |       |     |
        #            *-------*     |
        #            |     (2,-2)  |
        #           -y             |
        # ==========================
        roi1 = (0, 2), (2, 2), (2, 0), (0, 0)
        roi2 = (0, 0), (2, 0), (2, -2), (0, -2)
        polygon1 = Polygon(roi1)
        polygon2 = Polygon(roi2)
        result = intersection_polygon(polygon1, polygon2)
        self.assertEqual(0, len(result))


if __name__ == "__main__":
    main()

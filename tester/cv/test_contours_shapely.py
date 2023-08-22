# -*- coding: utf-8 -*-

from unittest import TestCase, main

from shapely.geometry import Polygon

from cvlayer.cv.contours_shapely import intersection_polygons


class ContoursShapelyTestCase(TestCase):
    def test_intersection_polygons(self):
        roi1 = (-1, 1), (1, 1), (1, -1), (-1, -1)
        roi2 = (0, 0), (2, 0), (2, -2), (0, -2)
        polygon1 = Polygon(roi1)
        polygon2 = Polygon(roi2)
        result = intersection_polygons(polygon1, polygon2)

        expected_bbox = [0, -1, 1, 0]
        result_bbox = list(round(i) for i in result.bounds)
        self.assertEqual(expected_bbox, result_bbox)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

from unittest import TestCase, main

from shapely.geometry import Polygon

from cvlayer.cv.contours_intersection import intersection_polygon_and_polygon


class ContoursIntersectionTestCase(TestCase):
    def test_intersection_polygon_polygon(self):
        # =============================
        #              +y             |
        #               |             |
        #               |             |
        #        (-1,1) |             |
        #           *-------*         |
        #           |   |   |         |
        # -x -------|---*---|---*--- +x
        #           |   |   |   |     |
        #           *-------*   |     |
        #               |       |     |
        #               *-------*     |
        #               |     (2,-2)  |
        #              -y             |
        # =============================
        ps1 = (-1, 1), (1, 1), (1, -1), (-1, -1)
        ps2 = (0, 0), (2, 0), (2, -2), (0, -2)
        polygon1 = Polygon(ps1)
        polygon2 = Polygon(ps2)
        self.assertEqual("Polygon", polygon1.intersection(polygon2).geom_type)
        result = intersection_polygon_and_polygon(polygon1, polygon2)
        self.assertEqual(1, len(result))
        self.assertEqual("Polygon", result[0].geom_type)

        self.assertTupleEqual((0.0, -1.0, 1.0, 0.0), result[0].bounds)

    def test_intersection_polygon_point(self):
        # =============================
        #              +y             |
        #    (-2,2)     |             |
        #       *-------*             |
        #       |       |             |
        #       |       |             |
        #       |       |             |
        # -x ---*-------*-------*--- +x
        #               |       |     |
        #               |       |     |
        #               |       |     |
        #               *-------*     |
        #               |     (2,-2)  |
        #              -y             |
        # =============================
        ps1 = (-2, 2), (0, 2), (0, 0), (-2, 0)
        ps2 = (0, 0), (2, 0), (2, -2), (0, -2)
        polygon1 = Polygon(ps1)
        polygon2 = Polygon(ps2)
        self.assertEqual("Point", polygon1.intersection(polygon2).geom_type)
        result = intersection_polygon_and_polygon(polygon1, polygon2)
        self.assertEqual(0, len(result))

    def test_intersection_polygon_multipoint(self):
        # =============================
        #              +y             |
        #    (-2,2)     |             |
        #       *---------------*     |
        #       |       | (1,1) |     |
        #       |       |   *   |     |
        #       |       | /   \ |     |
        # -x ---*-------*-------*--- +x
        #               |       |     |
        #               |       |     |
        #               |       |     |
        #               *-------*     |
        #               |     (2,-2)  |
        #              -y             |
        # =============================
        ps1 = (-2, 2), (2, 2), (2, 0), (1, 1), (0, 0), (-2, 0)
        ps2 = (0, 0), (2, 0), (2, -2), (0, -2)
        polygon1 = Polygon(ps1)
        polygon2 = Polygon(ps2)
        self.assertEqual("MultiPoint", polygon1.intersection(polygon2).geom_type)
        result = intersection_polygon_and_polygon(polygon1, polygon2)
        self.assertEqual(0, len(result))

    def test_intersection_polygon_linestring(self):
        # =============================
        #              +y             |
        #               |     (2,2)   |
        #               *-------*     |
        #               |       |     |
        #               |       |     |
        #               |       |     |
        # -x -----------*-------*--- +x
        #               |       |     |
        #               |       |     |
        #               |       |     |
        #               *-------*     |
        #               |     (2,-2)  |
        #              -y             |
        # =============================
        ps1 = (0, 2), (2, 2), (2, 0), (0, 0)
        ps2 = (0, 0), (2, 0), (2, -2), (0, -2)
        polygon1 = Polygon(ps1)
        polygon2 = Polygon(ps2)
        self.assertEqual("LineString", polygon1.intersection(polygon2).geom_type)
        result = intersection_polygon_and_polygon(polygon1, polygon2)
        self.assertEqual(0, len(result))

    def test_intersection_polygon_multipolygon(self):
        # ===============================
        #               +y              |
        #                |     (2,2)    |
        #        *---------------*      |
        #        |       | (1,1) |      |
        #        |   *-------*   |      |
        # (-3,0) |   |       |   |      |
        # -x *---|---|---+---|---|---* +x
        #    |   |[1]|   |   |[0]|   |  |
        #    |   *---*   |   *---*   |  |
        #    |           |    (2,-1) |  |
        #    *-----------------------*  |
        #                |         (3,-2)
        #               -y              |
        # ===============================
        ps1 = (-2, 2), (2, 2), (2, -1), (1, -1), (1, 1), (-1, 1), (-1, -1), (-2, -1)
        ps2 = (-3, 0), (3, 0), (3, -2), (-3, -2)
        polygon1 = Polygon(ps1)
        polygon2 = Polygon(ps2)
        self.assertEqual("MultiPolygon", polygon1.intersection(polygon2).geom_type)
        result = intersection_polygon_and_polygon(polygon1, polygon2)
        self.assertEqual(2, len(result))
        self.assertEqual("Polygon", result[0].geom_type)
        self.assertEqual("Polygon", result[1].geom_type)

        self.assertTupleEqual((1.0, -1.0, 2.0, 0.0), result[0].bounds)
        self.assertTupleEqual((-2.0, -1.0, -1.0, 0.0), result[1].bounds)


if __name__ == "__main__":
    main()

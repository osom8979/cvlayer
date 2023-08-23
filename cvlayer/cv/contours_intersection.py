# -*- coding: utf-8 -*-

from typing import List, Optional

from numpy.typing import NDArray
from shapely.geometry import Point, Polygon

from cvlayer.cv.cvt_shapely import (
    cvt_contour2linestring,
    cvt_contour2polygon,
    cvt_line2linestring,
    cvt_roi2polygon,
)
from cvlayer.geometry.flatten import flatten_line
from cvlayer.shape.points import raw_points
from cvlayer.shape.polygons import filter_polygons
from cvlayer.types import LineT, PointT, RectT


def intersection_polygon_and_polygon(
    polygon1: Polygon,
    polygon2: Polygon,
) -> List[Polygon]:
    return filter_polygons(polygon1.intersection(polygon2))


def intersection_roi_and_contour(roi: RectT, contour: NDArray) -> List[Polygon]:
    return intersection_polygon_and_polygon(
        cvt_contour2polygon(contour),
        cvt_roi2polygon(roi),
    )


def intersection_line_and_contour(line: LineT, contour: NDArray) -> List[PointT]:
    ls2 = cvt_line2linestring(*flatten_line(line))
    ls1 = cvt_contour2linestring(contour)
    return raw_points(ls1.intersection(ls2))


def intersection_line_and_line(line1: LineT, line2: LineT) -> Optional[PointT]:
    ls1 = cvt_line2linestring(*flatten_line(line1))
    ls2 = cvt_line2linestring(*flatten_line(line2))
    intersection = ls1.intersection(ls2)

    if intersection.is_empty:
        return None

    if intersection.geom_type == "LineString":
        raise ValueError("Probably the same line")

    if intersection.geom_type != "Point":
        raise TypeError(f"Unsupported geom type: {intersection.geom_type}")

    assert isinstance(intersection, Point)
    return intersection.x, intersection.y

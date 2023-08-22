# -*- coding: utf-8 -*-

from typing import List

from numpy import array, int32
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from cvlayer.types import Point


def convert_contour2polygon(contour: NDArray) -> Polygon:
    assert len(contour.shape) == 3
    assert contour.shape[0] >= 4
    assert contour.shape[1] == 1
    assert contour.shape[2] == 2
    return Polygon(contour[:, 0, :].tolist())


def convert_contour2linestring(contour: NDArray) -> LineString:
    assert len(contour.shape) == 3
    assert contour.shape[0] >= 4
    assert contour.shape[1] == 1
    assert contour.shape[2] == 2
    return LineString(contour[:, 0, :].tolist())


def convert_line2linestring(p1: Point, p2: Point) -> LineString:
    return LineString([[p1[0], p1[1]], [p2[0], p2[1]]])


def intersection_polygons(polygon1: Polygon, polygon2: Polygon) -> Polygon:
    intersection = polygon1.intersection(polygon2)
    if intersection.geom_type == "Polygon":
        assert isinstance(intersection, Polygon)
        return intersection
    else:
        union = unary_union(list(intersection))
        assert isinstance(union, (Polygon, MultiPolygon))


def geometry_to_contours(geom: BaseGeometry) -> List[NDArray]:
    if geom.geom_type == "Polygon":
        assert isinstance(geom, Polygon)
        exterior_coords = array(geom.exterior.coords, dtype=int32)
        return [exterior_coords]
    elif geom.geom_type == "MultiPolygon":
        assert isinstance(geom, MultiPolygon)
        contours = []
        for p in geom:
            exterior_coords = array(p.exterior.coords, dtype=int32)
            contours.append(exterior_coords)
        return contours
    else:
        raise ValueError("Unsupported geometry type")

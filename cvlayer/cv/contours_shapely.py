# -*- coding: utf-8 -*-

from itertools import chain
from typing import List, Optional

from numpy.typing import NDArray
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry


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


def convert_line2linestring(x1, y1, x2, y2) -> LineString:
    return LineString([[x1, y1], [x2, y2]])


def filter_polygons(base: Optional[BaseGeometry]) -> List[Polygon]:
    if base is None:
        return list()
    elif base.is_empty:
        return list()
    elif base.geom_type == "GeometryCollection":
        assert isinstance(base, GeometryCollection)
        return list(chain(*[filter_polygons(geom) for geom in base.geoms]))
    elif base.geom_type == "MultiPolygon":
        assert isinstance(base, MultiPolygon)
        return list(chain(*[filter_polygons(geom) for geom in base.geoms]))
    elif base.geom_type == "Polygon":
        assert isinstance(base, Polygon)
        return [base]
    elif base.geom_type == "MultiLineString":
        assert isinstance(base, MultiLineString)
        return list()
    elif base.geom_type == "LinearRing":
        assert isinstance(base, LinearRing)
        return list()
    elif base.geom_type == "LineString":
        assert isinstance(base, LineString)
        return list()
    elif base.geom_type == "MultiPoint":
        assert isinstance(base, MultiPoint)
        return list()
    elif base.geom_type == "Point":
        assert isinstance(base, Point)
        return list()
    else:
        raise TypeError(f"Unsupported geom type: {base.geom_type}")


def intersection_polygon(polygon1: Polygon, polygon2: Polygon) -> List[Polygon]:
    return filter_polygons(polygon1.intersection(polygon2))


# def geometry_to_contours(geom: BaseGeometry) -> List[NDArray]:
#     if geom.geom_type == "Polygon":
#         assert isinstance(geom, Polygon)
#         exterior_coords = array(geom.exterior.coords, dtype=int32)
#         return [exterior_coords]
#     elif geom.geom_type == "MultiPolygon":
#         assert isinstance(geom, MultiPolygon)
#         contours = []
#         for p in geom:
#             exterior_coords = array(p.exterior.coords, dtype=int32)
#             contours.append(exterior_coords)
#         return contours
#     else:
#         raise ValueError("Unsupported geometry type")

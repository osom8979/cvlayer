# -*- coding: utf-8 -*-

from itertools import chain
from typing import List, Optional

from shapely import (
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

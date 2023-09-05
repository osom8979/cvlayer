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

from cvlayer.typing import PointT


def raw_points(base: Optional[BaseGeometry]) -> List[PointT]:
    if base is None:
        return list()
    elif base.is_empty:
        return list()
    elif base.geom_type == "GeometryCollection":
        assert isinstance(base, GeometryCollection)
        return list(chain(*[raw_points(geom) for geom in base.geoms]))
    elif base.geom_type == "MultiPolygon":
        assert isinstance(base, MultiPolygon)
        raise NotImplementedError
    elif base.geom_type == "Polygon":
        assert isinstance(base, Polygon)
        raise NotImplementedError
    elif base.geom_type == "MultiLineString":
        assert isinstance(base, MultiLineString)
        return list(chain(*[raw_points(geom) for geom in base.geoms]))
    elif base.geom_type == "LinearRing":
        assert isinstance(base, LinearRing)
        raise NotImplementedError
    elif base.geom_type == "LineString":
        assert isinstance(base, LineString)
        return [(c[0], c[1]) for c in base.coords]
    elif base.geom_type == "MultiPoint":
        assert isinstance(base, MultiPoint)
        return list(chain(*[raw_points(geom) for geom in base.geoms]))
    elif base.geom_type == "Point":
        assert isinstance(base, Point)
        return [(base.x, base.y)]
    else:
        raise TypeError(f"Unsupported geom type: {base.geom_type}")

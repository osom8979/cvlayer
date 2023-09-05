# -*- coding: utf-8 -*-

from typing import List, Union

from numpy import array, int32, ndarray
from numpy.typing import NDArray
from shapely import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from cvlayer.typing import LineT, RectT


def cvt_contour2polygon(contour: NDArray) -> Polygon:
    assert len(contour.shape) == 3
    assert contour.shape[0] >= 4
    assert contour.shape[1] == 1
    assert contour.shape[2] == 2
    return Polygon(contour[:, 0, :].tolist())


def cvt_contour2linestring(contour: NDArray) -> LineString:
    assert len(contour.shape) == 3
    assert contour.shape[0] >= 4
    assert contour.shape[1] == 1
    assert contour.shape[2] == 2
    return LineString(contour[:, 0, :].tolist())


def cvt_line2linestring(line: LineT) -> LineString:
    return LineString([*line])


def cvt_polygon2contour(polygon: Polygon, dtype=int32) -> NDArray:
    if polygon.is_empty:
        raise ValueError("Polygon must not be empty")

    coords = polygon.exterior.coords
    shape = len(coords), 1, 2
    return array(coords, dtype=dtype).reshape(shape)


def cvt_multipolygon2contours(multipolygon: MultiPolygon, dtype=int32) -> List[NDArray]:
    if multipolygon.is_empty:
        raise ValueError("Polygon must not be empty")

    return [cvt_polygon2contour(polygon, dtype) for polygon in multipolygon.geoms]


def cvt_geometry2contours(geom: BaseGeometry, dtype=int32) -> List[NDArray]:
    if geom.is_empty:
        raise ValueError("Geometry must not be empty")

    if geom.geom_type == "Polygon":
        assert isinstance(geom, Polygon)
        return [cvt_polygon2contour(geom, dtype)]
    elif geom.geom_type == "MultiPolygon":
        assert isinstance(geom, MultiPolygon)
        return cvt_multipolygon2contours(geom, dtype)
    else:
        raise TypeError(f"Unsupported geom type: {geom.geom_type}")


def cvt_roi2contour(roi: RectT, dtype=int32) -> NDArray:
    x1, y1, x2, y2 = roi
    shell = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    raw = [p for xy in shell for p in xy]
    contour = array(raw, dtype=dtype).reshape((5, 1, 2))

    assert len(contour.shape) == 3
    assert contour.shape[0] == len(shell)
    assert contour.shape[1] == 1
    assert contour.shape[2] == 2

    return contour


def cvt_roi2polygon(roi: RectT) -> Polygon:
    x1, y1, x2, y2 = roi
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


AreaType = Union[NDArray, RectT]


def cvt_area2polygon(area: AreaType) -> Polygon:
    if isinstance(area, ndarray):
        return cvt_contour2polygon(area)
    elif isinstance(area, (tuple, list)):
        if len(area) == 4:
            return cvt_roi2polygon(area)
    raise TypeError(f"Unsupported area type: {type(area).__name__}")


class CvlShapely:
    @staticmethod
    def cvl_cvt_contour2polygon(contour: NDArray):
        return cvt_contour2polygon(contour)

    @staticmethod
    def cvl_cvt_contour2linestring(contour: NDArray):
        return cvt_contour2linestring(contour)

    @staticmethod
    def cvl_cvt_line2linestring(line: LineT):
        return cvt_line2linestring(line)

    @staticmethod
    def cvl_cvt_polygon2contour(polygon: Polygon, dtype=int32):
        return cvt_polygon2contour(polygon, dtype)

    @staticmethod
    def cvl_cvt_multipolygon2contours(multipolygon: MultiPolygon, dtype=int32):
        return cvt_multipolygon2contours(multipolygon, dtype)

    @staticmethod
    def cvl_cvt_geometry2contours(geom: BaseGeometry, dtype=int32):
        return cvt_geometry2contours(geom, dtype)

    @staticmethod
    def cvl_cvt_roi2contour(roi: RectT, dtype=int32):
        return cvt_roi2contour(roi, dtype)

    @staticmethod
    def cvl_cvt_roi2polygon(roi: RectT):
        return cvt_roi2polygon(roi)

    @staticmethod
    def cvl_cvt_area2polygon(area: AreaType):
        return cvt_area2polygon(area)

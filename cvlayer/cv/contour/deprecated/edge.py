# -*- coding: utf-8 -*-

from enum import Enum, auto, unique
from typing import Callable, List, NamedTuple, Tuple

from numpy import int32, ndarray
from numpy.typing import NDArray

from cvlayer.typing import PolygonT


@unique
class FindContourEdgeMethod(Enum):
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()


class ContourEdgePoints(NamedTuple):
    score: int
    contour: NDArray
    points: PolygonT


def _find_edge(
    filter_callable: Callable[[NDArray], NDArray],
    contour: NDArray,
) -> PolygonT:
    points = filter_callable(contour)[:, 0, :].tolist()
    assert isinstance(points, list)
    return [(p[0], p[1]) for p in points] if points else []


def _left_edge_filter(contour: NDArray) -> NDArray:
    value = contour[:, 0, 0].min()  # noqa
    return contour[contour[:, 0, 0] == value]


def _right_edge_filter(contour: NDArray) -> NDArray:
    value = contour[:, 0, 0].max()  # noqa
    return contour[contour[:, 0, 0] == value]


def _top_edge_filter(contour: NDArray) -> NDArray:
    value = contour[:, 0, 1].min()  # noqa
    return contour[contour[:, 0, 1] == value]


def _bottom_edge_filter(contour: NDArray) -> NDArray:
    value = contour[:, 0, 1].max()  # noqa
    return contour[contour[:, 0, 1] == value]


def find_left_edge(contour: NDArray) -> PolygonT:
    return _find_edge(_left_edge_filter, contour)


def find_right_edge(contour: NDArray) -> PolygonT:
    return _find_edge(_right_edge_filter, contour)


def find_top_edge(contour: NDArray) -> PolygonT:
    return _find_edge(_top_edge_filter, contour)


def find_bottom_edge(contour: NDArray) -> PolygonT:
    return _find_edge(_bottom_edge_filter, contour)


def find_edge_points(
    method: FindContourEdgeMethod,
    contour: NDArray,
) -> PolygonT:
    assert isinstance(contour, ndarray)
    assert contour.dtype == int32
    assert len(contour.shape) == 3
    assert contour.shape[0] >= 1
    assert contour.shape[1] == 1
    assert contour.shape[2] == 2

    if method == FindContourEdgeMethod.LEFT:
        return find_left_edge(contour)
    elif method == FindContourEdgeMethod.RIGHT:
        return find_right_edge(contour)
    elif method == FindContourEdgeMethod.TOP:
        return find_top_edge(contour)
    elif method == FindContourEdgeMethod.BOTTOM:
        return find_bottom_edge(contour)
    else:
        assert False, "Inaccessible section"


def _validate_edge_scores(
    method: FindContourEdgeMethod,
    edge_points: PolygonT,
) -> None:
    if not edge_points:
        return

    if method in (FindContourEdgeMethod.LEFT, FindContourEdgeMethod.RIGHT):
        score = edge_points[0][0]
        same_scores = list(filter(lambda p: p[0] == score, edge_points))
        assert len(same_scores) == len(edge_points)
    elif method in (FindContourEdgeMethod.TOP, FindContourEdgeMethod.BOTTOM):
        score = edge_points[0][1]
        same_scores = list(filter(lambda p: p[1] == score, edge_points))
        assert len(same_scores) == len(edge_points)
    else:
        assert False, "Inaccessible section"


def get_edge_score(
    method: FindContourEdgeMethod,
    edge_points: PolygonT,
) -> int:
    _validate_edge_scores(method, edge_points)

    if method in (FindContourEdgeMethod.LEFT, FindContourEdgeMethod.RIGHT):
        return edge_points[0][0]
    elif method in (FindContourEdgeMethod.TOP, FindContourEdgeMethod.BOTTOM):
        return edge_points[0][1]
    else:
        assert False, "Inaccessible section"


def find_best_contour_edge_points(
    method: FindContourEdgeMethod,
    contours: List[NDArray],
) -> ContourEdgePoints:
    contours_edge_points = [find_edge_points(method, contour) for contour in contours]

    scores: List[Tuple[int, int]] = list()  # List[Tuple[index, score]]
    for i, contour_edge_points in enumerate(contours_edge_points):
        score = get_edge_score(method, contour_edge_points)
        scores.append((i, score))

    if method in (FindContourEdgeMethod.LEFT, FindContourEdgeMethod.TOP):
        score_elem = min(scores, key=lambda x: x[1])
    elif method in (FindContourEdgeMethod.RIGHT, FindContourEdgeMethod.BOTTOM):
        score_elem = max(scores, key=lambda x: x[1])
    else:
        assert False, "Inaccessible section"

    index = score_elem[0]
    score = score_elem[1]
    contour = contours[index]
    points = contours_edge_points[index]
    return ContourEdgePoints(score, contour, points)


def find_leftmost_contour(contours: List[NDArray]):
    return find_best_contour_edge_points(FindContourEdgeMethod.LEFT, contours)


def find_rightmost_contour(contours: List[NDArray]):
    return find_best_contour_edge_points(FindContourEdgeMethod.RIGHT, contours)


def find_topmost_contour(contours: List[NDArray]):
    return find_best_contour_edge_points(FindContourEdgeMethod.TOP, contours)


def find_bottommost_contour(contours: List[NDArray]):
    return find_best_contour_edge_points(FindContourEdgeMethod.BOTTOM, contours)

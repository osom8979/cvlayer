# -*- coding: utf-8 -*-
# https://en.wikipedia.org/wiki/Quadrant_(plane_geometry)

from typing import Tuple

from numpy import logical_and, sqrt
from numpy.typing import NDArray

from cvlayer.typing import PointF


def farthest_points_of_quadrants(
    contour: NDArray,
    center: PointF,
) -> Tuple[PointF, PointF, PointF, PointF]:
    contour_shape = contour.shape
    assert len(contour_shape) == 3
    assert 1 == contour_shape[1]
    assert 2 == contour_shape[2]

    cx, cy = center

    q1 = contour[logical_and(contour[:, 0, 0] > cx, contour[:, 0, 1] > cy)]
    q2 = contour[logical_and(contour[:, 0, 0] < cx, contour[:, 0, 1] > cy)]
    q3 = contour[logical_and(contour[:, 0, 0] < cx, contour[:, 0, 1] < cy)]
    q4 = contour[logical_and(contour[:, 0, 0] > cx, contour[:, 0, 1] < cy)]

    q1d = sqrt((q1[:, 0, 0] - cx) ** 2 + (q1[:, 0, 1] - cy) ** 2)
    q2d = sqrt((q2[:, 0, 0] - cx) ** 2 + (q2[:, 0, 1] - cy) ** 2)
    q3d = sqrt((q3[:, 0, 0] - cx) ** 2 + (q3[:, 0, 1] - cy) ** 2)
    q4d = sqrt((q4[:, 0, 0] - cx) ** 2 + (q4[:, 0, 1] - cy) ** 2)

    q1p = q1[q1d.argmax()][0]
    q2p = q2[q2d.argmax()][0]
    q3p = q3[q3d.argmax()][0]
    q4p = q4[q4d.argmax()][0]

    p1 = float(q1p[0]), float(q1p[1])
    p2 = float(q2p[0]), float(q2p[1])
    p3 = float(q3p[0]), float(q3p[1])
    p4 = float(q4p[0]), float(q4p[1])

    return p1, p2, p3, p4

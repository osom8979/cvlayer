# -*- coding: utf-8 -*-

from typing import Final, Optional

from numpy import int32, ndarray, uint8, zeros
from numpy.typing import NDArray

from cvlayer.cv.color import PIXEL_8BIT_MAX
from cvlayer.cv.contour.analysis import contour_area
from cvlayer.cv.contour.find import find_contours
from cvlayer.cv.drawable.contours import draw_contour
from cvlayer.cv.types.chain_approx import DEFAULT_CHAIN_APPROX
from cvlayer.cv.types.line_type import LINE_8
from cvlayer.cv.types.retrieval import DEFAULT_RETRIEVAL
from cvlayer.cv.types.thickness import FILLED
from cvlayer.typing import SizeI

DISABLE_AREA_FILTER: Final[float] = -1.0


class LargestContourResult:
    def __init__(
        self,
        frame_size: SizeI,
        contour: Optional[NDArray[int32]] = None,
        mask: Optional[NDArray[uint8]] = None,
        area: Optional[float] = None,
        mask_value=PIXEL_8BIT_MAX,
        area_oriented=False,
    ):
        self._frame_size = frame_size
        self._contour = contour
        self._mask = mask
        self._area = area
        self._mask_value = mask_value
        self._area_oriented = area_oriented

    @property
    def frame_size(self) -> SizeI:
        return self._frame_size

    @property
    def mask_value(self) -> int:
        return self._mask_value

    @property
    def area_oriented(self) -> bool:
        return self._area_oriented

    @property
    def has_contour(self) -> bool:
        return self._contour is not None

    @property
    def contour(self):
        return self._contour

    @property
    def mask(self) -> NDArray[uint8]:
        if self._mask is None:
            if self._contour is None:
                raise ValueError("Contour does not exist")
            w, h = self._frame_size
            mask = zeros((h, w), dtype=uint8)
            self._mask = draw_contour(
                image=mask,
                contour=self._contour,
                color=self._mask_value,
                thickness=FILLED,
                line=LINE_8,
            )
        assert self._mask is not None
        return self._mask

    @property
    def area(self) -> float:
        if self._area is None:
            if self._contour is None:
                raise ValueError("Contour does not exist")
            self._area = contour_area(self._contour, self._area_oriented)
        assert self._area is not None
        return self._area


def find_contours_filter_area_largest(
    image: NDArray,
    mode=DEFAULT_RETRIEVAL,
    method=DEFAULT_CHAIN_APPROX,
    area_oriented=False,
    area_min=DISABLE_AREA_FILTER,
    area_max=DISABLE_AREA_FILTER,
    mask_value=PIXEL_8BIT_MAX,
) -> LargestContourResult:
    contours = find_contours(image, mode, method).contours
    h, w = image.shape[0:2]
    image_size = w, h

    if len(contours) == 0:
        return LargestContourResult(
            frame_size=image_size,
            contour=None,
            mask=None,
            area=None,
            mask_value=mask_value,
            area_oriented=area_oriented,
        )

    largest_contour: Optional[NDArray[int32]] = None
    largest_area = 0.0

    # [IMPORTANT]
    # 'map object' is not an iterator in 'zip'. So convert it to 'list'.
    areas = list(map(lambda c: contour_area(c, area_oriented), contours))

    for contour, area in zip(contours, areas):
        assert isinstance(contour, ndarray)
        assert isinstance(area, float)

        if area_min >= 0.0 and area < area_min:
            continue
        if area_max >= 0.0 and area_max < area:  # noqa
            continue

        if largest_contour is None or largest_area < area:
            largest_contour = contour
            largest_area = area

    return LargestContourResult(
        frame_size=image_size,
        contour=largest_contour,
        mask=None,
        area=largest_area,
        mask_value=mask_value,
        area_oriented=area_oriented,
    )

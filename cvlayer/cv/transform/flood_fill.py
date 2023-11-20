# -*- coding: utf-8 -*-

from typing import Final, NamedTuple, Optional, Union

import cv2
from numpy import float16, float32, float64, int8, ndarray, uint8, zeros
from numpy.typing import NDArray

from cvlayer.cv.types.color import ColorLike, normalize_color
from cvlayer.cv.types.connectivity import DEFAULT_CONNECTIVITY, normalize_connectivity
from cvlayer.typing import PointI, RectI

FLOODFILL_FIXED_RANGE: Final[int] = cv2.FLOODFILL_FIXED_RANGE
"""
If set, the difference between the current pixel and seed pixel is considered.
Otherwise, the difference between neighbor pixels is considered
(that is, the range is floating).
"""

FLOODFILL_MASK_ONLY: Final[int] = cv2.FLOODFILL_MASK_ONLY
"""
If set, the function does not change the image (newVal is ignored), and
only fills the mask with the value specified in bits 8-16 of flags as described above.
This option only make sense in function variants that have the mask parameter.
"""

assert FLOODFILL_FIXED_RANGE == 65_536 == 0b1_0000_0000_0000_0000 == 1 << 16
assert FLOODFILL_MASK_ONLY == 131_072 == 0b10_0000_0000_0000_0000 == 1 << 17


class FloodFillFlag:
    def __init__(
        self,
        connectivity=DEFAULT_CONNECTIVITY,
        fill=1,
        fixed_range=False,
        mask_only=False,
    ):
        self.connectivity = normalize_connectivity(connectivity)
        assert self.connectivity in (4, 8)
        self.fill = fill
        assert 1 <= self.fill <= 255
        self.fixed_range = fixed_range
        self.mask_only = mask_only

    @classmethod
    def from_flags(cls, flags: int):
        connectivity = flags & 0xFF
        fill = (flags >> 8) & 0xFF
        fixed_range = (flags & FLOODFILL_FIXED_RANGE) == FLOODFILL_FIXED_RANGE
        mask_only = (flags & FLOODFILL_MASK_ONLY) == FLOODFILL_MASK_ONLY
        return cls(connectivity, fill, fixed_range, mask_only)

    def as_flags(self) -> int:
        result = self.connectivity
        result |= self.fill << 8
        if self.fixed_range:
            result |= FLOODFILL_FIXED_RANGE
        if self.mask_only:
            result |= FLOODFILL_MASK_ONLY
        return result

    def __int__(self) -> int:
        return self.as_flags()


FloodFillFlagLike = Union[FloodFillFlag, int]
DEFAULT_FLOOD_FILL_FLAG: Final[FloodFillFlagLike] = 4


def normalize_flood_fill_flag(flags: FloodFillFlagLike) -> int:
    if isinstance(flags, FloodFillFlag):
        return flags.as_flags()
    elif isinstance(flags, int):
        return flags
    else:
        raise TypeError(f"Unsupported flood-fill flags type: {type(flags).__name__}")


class FloodFillResult(NamedTuple):
    number_of_filled_pixels: int
    image: NDArray
    mask: NDArray
    roi: RectI


def flood_fill(
    image: NDArray,
    mask: Optional[NDArray],
    seed_point: PointI,
    new_value: ColorLike,
    lower_diff: Optional[ColorLike] = None,
    upper_diff: Optional[ColorLike] = None,
    flags=DEFAULT_FLOOD_FILL_FLAG,
) -> FloodFillResult:
    is_img_1c = len(image.shape) == 2
    is_img_3c = len(image.shape) == 3 and image.shape[2] == 3
    if not is_img_1c and not is_img_3c:
        raise ValueError(f"The image argument must be 1 or 3 channels: {image.shape}")
    if image.dtype not in (int8, uint8, float16, float32, float64):
        raise TypeError(f"The image arg must be 8-bit or floating-point: {image.dtype}")

    image_height, image_width = image.shape[0:2]

    # If an empty mask is passed it will be created automatically.
    if mask is None:
        mask = zeros((image_height + 2, image_width + 2), dtype=uint8)

    assert isinstance(mask, ndarray)

    if len(mask.shape) != 2:
        raise ValueError(f"The mask argument must be single-channels: {mask.shape}")
    if mask.dtype not in (int8, uint8):
        raise TypeError(f"The mask argument must be 8-bits: {mask.dtype}")

    mask_height, mask_width = mask.shape

    # Additionally, the function fills the border of the mask with ones to simplify
    # internal processing.
    if image_height + 2 != mask_height:
        raise ValueError("2 pixels taller than image")
    if image_width + 2 != mask_width:
        raise ValueError("2 pixels wider than image")

    _new_value = normalize_color(new_value)
    _lower_diff = normalize_color(lower_diff) if lower_diff is not None else tuple()
    _upper_diff = normalize_color(upper_diff) if upper_diff is not None else tuple()
    _flags = normalize_flood_fill_flag(flags)

    result = cv2.floodFill(
        image,
        mask,
        seed_point,
        _new_value,
        _lower_diff,
        _upper_diff,
        _flags,
    )

    ret_count, ret_image, ret_mask, ret_rect = result
    # Since the mask is larger than the filled image,
    # a pixel (x,y) in image corresponds to the pixel (x+1,y+1) in the mask.

    x1, y1, x2, y2 = ret_rect

    return FloodFillResult(ret_count, ret_image, ret_mask, (x1, y1, x2, y2))


class CvlTransformFloodFill:
    @staticmethod
    def cvl_create_flood_fill_flag(
        connectivity=DEFAULT_CONNECTIVITY,
        fill=1,
        fixed_range=False,
        mask_only=False,
    ):
        return FloodFillFlag(connectivity, fill, fixed_range, mask_only)

    @staticmethod
    def cvl_normalize_flood_fill_flag(flags: FloodFillFlagLike) -> int:
        return normalize_flood_fill_flag(flags)

    @staticmethod
    def cvl_flood_fill(
        image: NDArray,
        mask: Optional[NDArray],
        seed_point: PointI,
        new_value: ColorLike,
        lower_diff: Optional[ColorLike] = None,
        upper_diff: Optional[ColorLike] = None,
        flags=DEFAULT_FLOOD_FILL_FLAG,
    ):
        return flood_fill(
            image,
            mask,
            seed_point,
            new_value,
            lower_diff,
            upper_diff,
            flags,
        )

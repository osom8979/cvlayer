# -*- coding: utf-8 -*-

from typing import Optional, Sequence

import cv2
from numpy.typing import NDArray


def image_write(
    filename: str,
    image: NDArray,
    params: Optional[Sequence[int]] = None,
) -> bool:
    if params:
        return cv2.imwrite(filename, image, params)
    else:
        return cv2.imwrite(filename, image)


def image_read(filename: str, flags: int) -> NDArray:
    return cv2.imread(filename, flags)


class CvlImageIo:
    @staticmethod
    def cvl_image_write(
        filename: str,
        image: NDArray,
        params: Optional[Sequence[int]] = None,
    ) -> bool:
        return image_write(filename, image, params)

    @staticmethod
    def cvl_image_read(filename: str, flags: int) -> NDArray:
        return image_read(filename, flags)

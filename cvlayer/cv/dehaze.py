# -*- coding: utf-8 -*-

from math import floor
from typing import Final, Tuple

import cv2
from numpy import empty, float64, zeros
from numpy.typing import NDArray

from cvlayer.cv.types.data_type import CV_64F

DEFAULT_DEHAZE_KERNEL_SIZE: Final[Tuple[int, int]] = 15, 15


def darkest_channel(frame: NDArray, kernel: NDArray) -> NDArray:
    assert frame.dtype == float64
    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]
    darkest = cv2.min(cv2.min(r, g), b)
    return cv2.erode(darkest, kernel)


def atmospheric_light(image: NDArray, dark: NDArray) -> NDArray:
    h, w = image.shape[:2]
    area = h * w

    num_px = int(max(floor(area / 1000), 1))
    dark_vec = dark.reshape(area)
    im_vec = image.reshape(area, 3)

    begin = area - num_px

    indices = dark_vec.argsort()
    indices = indices[begin::]

    atm_sum = zeros([1, 3])
    for ind in range(1, num_px):
        atm_sum = atm_sum + im_vec[indices[ind]]

    return atm_sum / num_px


def transmission_estimate(image: NDArray, a, kernel: NDArray):
    omega = 0.95
    im3 = empty(image.shape, image.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = image[:, :, ind] / a[0, ind]

    return 1 - omega * darkest_channel(im3, kernel)


def guided_filter(image, p, r, eps):
    mean_i = cv2.boxFilter(image, CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, CV_64F, (r, r))
    mean_ip = cv2.boxFilter(image * p, CV_64F, (r, r))
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = cv2.boxFilter(image * image, CV_64F, (r, r))
    var_i = mean_ii - mean_i * mean_i

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, CV_64F, (r, r))

    return mean_a * image + mean_b


def transmission_refine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = float64(gray) / 255
    r = 60
    eps = 0.0001
    return guided_filter(gray, et, r, eps)


def recover(im, t, a, tx=0.1) -> NDArray:
    res = empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - a[0, ind]) / t + a[0, ind]

    return res


def dehaze(frame: NDArray, kernel: NDArray) -> NDArray:
    floating_frame = frame.astype(dtype=float64) / 255
    dark = darkest_channel(floating_frame, kernel)
    light = atmospheric_light(floating_frame, dark)
    te = transmission_estimate(floating_frame, light, kernel)
    tr = transmission_refine(frame, te)
    return recover(floating_frame, tr, light, 0.1)


class Dehaze:
    def __init__(self, kernel_size=DEFAULT_DEHAZE_KERNEL_SIZE):
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    def run(self, frame: NDArray) -> NDArray:
        return dehaze(frame, self._kernel)


class CvlDehaze:
    @staticmethod
    def cvl_create_dehaze(kernel_size=DEFAULT_DEHAZE_KERNEL_SIZE):
        return Dehaze(kernel_size)

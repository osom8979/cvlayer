# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Tuple, Union

import cv2
from numpy.typing import NDArray


@unique
class VideoCaptureProperty(Enum):
    POS_MSEC = cv2.CAP_PROP_POS_MSEC
    POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
    POS_AVI_RATIO = cv2.CAP_PROP_POS_AVI_RATIO
    FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
    FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
    FPS = cv2.CAP_PROP_FPS
    FOURCC = cv2.CAP_PROP_FOURCC
    FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    FORMAT = cv2.CAP_PROP_FORMAT
    MODE = cv2.CAP_PROP_MODE
    BRIGHTNESS = cv2.CAP_PROP_BRIGHTNESS
    CONTRAST = cv2.CAP_PROP_CONTRAST
    SATURATION = cv2.CAP_PROP_SATURATION
    HUE = cv2.CAP_PROP_HUE
    GAIN = cv2.CAP_PROP_GAIN
    EXPOSURE = cv2.CAP_PROP_EXPOSURE
    CONVERT_RGB = cv2.CAP_PROP_CONVERT_RGB
    WHITE_BALANCE_BLUE_U = cv2.CAP_PROP_WHITE_BALANCE_BLUE_U
    RECTIFICATION = cv2.CAP_PROP_RECTIFICATION
    MONOCHROME = cv2.CAP_PROP_MONOCHROME
    SHARPNESS = cv2.CAP_PROP_SHARPNESS
    AUTO_EXPOSURE = cv2.CAP_PROP_AUTO_EXPOSURE
    GAMMA = cv2.CAP_PROP_GAMMA
    TEMPERATURE = cv2.CAP_PROP_TEMPERATURE
    TRIGGER = cv2.CAP_PROP_TRIGGER
    TRIGGER_DELAY = cv2.CAP_PROP_TRIGGER_DELAY
    WHITE_BALANCE_RED_V = cv2.CAP_PROP_WHITE_BALANCE_RED_V
    ZOOM = cv2.CAP_PROP_ZOOM
    FOCUS = cv2.CAP_PROP_FOCUS
    GUID = cv2.CAP_PROP_GUID
    ISO_SPEED = cv2.CAP_PROP_ISO_SPEED
    BACKLIGHT = cv2.CAP_PROP_BACKLIGHT
    PAN = cv2.CAP_PROP_PAN
    TILT = cv2.CAP_PROP_TILT
    ROLL = cv2.CAP_PROP_ROLL
    IRIS = cv2.CAP_PROP_IRIS
    SETTINGS = cv2.CAP_PROP_SETTINGS
    BUFFERSIZE = cv2.CAP_PROP_BUFFERSIZE
    AUTOFOCUS = cv2.CAP_PROP_AUTOFOCUS
    SAR_NUM = cv2.CAP_PROP_SAR_NUM
    SAR_DEN = cv2.CAP_PROP_SAR_DEN
    BACKEND = cv2.CAP_PROP_BACKEND
    CHANNEL = cv2.CAP_PROP_CHANNEL
    AUTO_WB = cv2.CAP_PROP_AUTO_WB
    WB_TEMPERATURE = cv2.CAP_PROP_WB_TEMPERATURE
    CODEC_PIXEL_FORMAT = cv2.CAP_PROP_CODEC_PIXEL_FORMAT
    BITRATE = cv2.CAP_PROP_BITRATE
    ORIENTATION_META = cv2.CAP_PROP_ORIENTATION_META
    ORIENTATION_AUTO = cv2.CAP_PROP_ORIENTATION_AUTO
    OPEN_TIMEOUT_MSEC = cv2.CAP_PROP_OPEN_TIMEOUT_MSEC
    READ_TIMEOUT_MSEC = cv2.CAP_PROP_READ_TIMEOUT_MSEC


class VideoCapture:
    def __init__(self, file: Union[str, int]):
        self._cap = cv2.VideoCapture(file)

    @property
    def opened(self) -> bool:
        return self._cap.isOpened()

    @property
    def backend_name(self) -> str:
        return self._cap.getBackendName()

    @property
    def exception_mode(self) -> bool:
        return self._cap.getExceptionMode()

    @exception_mode.setter
    def exception_mode(self, enable: bool) -> None:
        self._cap.setExceptionMode(enable)

    def release(self) -> None:
        self._cap.release()

    def grab(self) -> bool:
        return self._cap.grab()

    def retrieve(self, flag=0) -> Tuple[bool, NDArray]:
        return self._cap.retrieve(None, flag)

    def read(self) -> Tuple[bool, NDArray]:
        return self._cap.read()

    def set_property(self, prop: VideoCaptureProperty, value: float) -> bool:
        return self._cap.set(prop.value, value)

    def get_property(self, prop: VideoCaptureProperty) -> float:
        return self._cap.get(prop.value)

    @property
    def width(self) -> int:
        return int(self.get_property(VideoCaptureProperty.FRAME_WIDTH))

    @width.setter
    def width(self, value: int) -> None:
        self.set_property(VideoCaptureProperty.FRAME_WIDTH, float(value))

    @property
    def height(self) -> int:
        return int(self.get_property(VideoCaptureProperty.FRAME_HEIGHT))

    @height.setter
    def height(self, value: int) -> None:
        self.set_property(VideoCaptureProperty.FRAME_HEIGHT, value)

    @property
    def fps(self) -> float:
        return self.get_property(VideoCaptureProperty.FPS)

    @fps.setter
    def fps(self, value: float) -> None:
        self.set_property(VideoCaptureProperty.FPS, value)

    @property
    def frames(self) -> int:
        return int(self.get_property(VideoCaptureProperty.FRAME_COUNT))

    @frames.setter
    def frames(self, value: int) -> None:
        self.set_property(VideoCaptureProperty.FRAME_COUNT, float(value))

    @property
    def pos(self) -> int:
        return int(self.get_property(VideoCaptureProperty.POS_FRAMES))

    @pos.setter
    def pos(self, value: int) -> None:
        self.set_property(VideoCaptureProperty.POS_FRAMES, float(value))

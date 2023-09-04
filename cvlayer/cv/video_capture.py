# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Optional, Sequence, Tuple, Union

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


class CaptureDomain(Enum):
    ANY = cv2.CAP_ANY  # Auto detect == 0.
    VFW = cv2.CAP_VFW  # Video For Windows (platform native)
    V4L = cv2.CAP_V4L  # V4L/V4L2 capturing support via libv4l.
    V4L2 = cv2.CAP_V4L2  # Same as CAP_V4L.
    FIREWIRE = cv2.CAP_FIREWIRE  # IEEE 1394 drivers.
    FIREWARE = cv2.CAP_FIREWARE  # Same as CAP_FIREWIRE.
    IEEE1394 = cv2.CAP_IEEE1394  # Same as CAP_FIREWIRE.
    DC1394 = cv2.CAP_DC1394  # Same as CAP_FIREWIRE.
    CMU1394 = cv2.CAP_CMU1394  # Same as CAP_FIREWIRE.
    QT = cv2.CAP_QT  # QuickTime.
    UNICAP = cv2.CAP_UNICAP  # Unicap drivers.
    DSHOW = cv2.CAP_DSHOW  # DirectShow (via videoInput)
    PVAPI = cv2.CAP_PVAPI  # PvAPI, Prosilica GigE SDK.
    OPENNI = cv2.CAP_OPENNI  # OpenNI (for Kinect)
    OPENNI_ASUS = cv2.CAP_OPENNI_ASUS  # OpenNI (for Asus Xtion)
    ANDROID = cv2.CAP_ANDROID  # Android - not used.
    XIAPI = cv2.CAP_XIAPI  # XIMEA Camera API.
    AVFOUNDATION = cv2.CAP_AVFOUNDATION  # AVFoundation framework for iOS
    """OS X Lion will have the same API"""

    GIGANETIX = cv2.CAP_GIGANETIX  # Smartek Giganetix GigEVisionSDK.
    MSMF = cv2.CAP_MSMF  # Microsoft Media Foundation (via videoInput)
    WINRT = cv2.CAP_WINRT  # Microsoft Windows Runtime using Media Foundation.
    INTELPERC = cv2.CAP_INTELPERC  # Intel Perceptual Computing SDK.
    OPENNI2 = cv2.CAP_OPENNI2  # OpenNI2 (for Kinect)
    OPENNI2_ASUS = cv2.CAP_OPENNI2_ASUS  # OpenNI2
    """for Asus Xtion and Occipital Structure sensors"""

    GPHOTO2 = cv2.CAP_GPHOTO2  # gPhoto2 connection
    GSTREAMER = cv2.CAP_GSTREAMER  # GStreamer.
    FFMPEG = cv2.CAP_FFMPEG  # Open and record video file or stream using the FFMPEG
    IMAGES = cv2.CAP_IMAGES  # OpenCV Image Sequence (e.g. img_%02d.jpg)
    ARAVIS = cv2.CAP_ARAVIS  # Aravis SDK.
    OPENCV_MJPEG = cv2.CAP_OPENCV_MJPEG  # Built-in OpenCV MotionJPEG codec.
    INTEL_MFX = cv2.CAP_INTEL_MFX  # Intel MediaSDK.
    XINE = cv2.CAP_XINE  # XINE engine (Linux)


def get_capture_index(index: int, domain: CaptureDomain) -> int:
    return index + domain.value


class VideoCapture:
    def __init__(
        self,
        file: Optional[Union[int, str]] = None,
        api: Optional[int] = None,
        params: Optional[Sequence[int]] = None,
    ):
        if file is not None:
            if api is not None:
                if params is not None:
                    self._capture = cv2.VideoCapture(file, api, params)
                else:
                    self._capture = cv2.VideoCapture(file, api)
            else:
                self._capture = cv2.VideoCapture(file)
        else:
            self._capture = cv2.VideoCapture()

    @staticmethod
    def get_auto_detect_index(index: int) -> int:
        return get_capture_index(index, CaptureDomain.ANY)

    @staticmethod
    def get_ffmpeg_index(index: int) -> int:
        return get_capture_index(index, CaptureDomain.FFMPEG)

    @staticmethod
    def get_images_index(index: int) -> int:
        return get_capture_index(index, CaptureDomain.IMAGES)

    @staticmethod
    def get_dshow_index(index: int) -> int:
        return get_capture_index(index, CaptureDomain.DSHOW)

    @staticmethod
    def get_msmf_index(index: int) -> int:
        return get_capture_index(index, CaptureDomain.MSMF)

    @staticmethod
    def get_v4l2_index(index: int) -> int:
        return get_capture_index(index, CaptureDomain.V4L2)

    @property
    def opened(self) -> bool:
        return self._capture.isOpened()

    @property
    def capture(self) -> cv2.VideoCapture:
        return self._capture

    def open(
        self,
        file: Union[int, str],
        api: Optional[int] = None,
        params: Optional[Sequence[int]] = None,
    ) -> bool:
        if api is not None:
            if params is not None:
                return self._capture.open(file, apiPreference=api, params=params)
            else:
                return self._capture.open(file, apiPreference=api)
        else:
            return self._capture.open(file)

    def release(self) -> None:
        return self._capture.release()

    def grab(self) -> bool:
        return self._capture.grab()

    def retrieve(
        self,
        image: Optional[NDArray] = None,
        flag: Optional[int] = None,
    ) -> Tuple[bool, NDArray]:
        if flag is not None:
            return self._capture.retrieve(image, flag)
        else:
            return self._capture.retrieve(image)

    def read(self, image: Optional[NDArray] = None) -> Tuple[bool, NDArray]:
        return self._capture.read(image)

    def set(self, prop: int, value: float) -> bool:
        return self._capture.set(prop, value)

    def get(self, prop: int) -> float:
        return self._capture.get(prop)

    def set_property(self, prop: VideoCaptureProperty, value: float) -> bool:
        return self.set(prop.value, value)

    def get_property(self, prop: VideoCaptureProperty) -> float:
        return self.get(prop.value)

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

    def get_backend_name(self) -> str:
        return self._capture.getBackendName()

    def set_exception_mode(self, enable: bool) -> None:
        self._capture.setExceptionMode(enable)

    def get_exception_mode(self) -> bool:
        return self._capture.getExceptionMode()

    @staticmethod
    def wait_any(
        streams: Sequence[Union["VideoCapture", cv2.VideoCapture]],
        timeout_nano: Optional[int] = None,
    ) -> Tuple[bool, Sequence[int]]:
        cv2_streams = list()
        for stream in streams:
            if isinstance(stream, cv2.VideoCapture):
                cv2_streams.append(stream)
            elif isinstance(stream, VideoCapture):
                cv2_streams.append(stream.capture)
        if timeout_nano is not None:
            return cv2.VideoCapture.waitAny(cv2_streams, timeoutNs=timeout_nano)
        else:
            return cv2.VideoCapture.waitAny(cv2_streams)


class CvlVideoCapture:
    @staticmethod
    def cvl_create_video_capture(
        file: Optional[Union[int, str]] = None,
        api: Optional[int] = None,
        params: Optional[Sequence[int]] = None,
    ) -> VideoCapture:
        return VideoCapture(file, api, params)

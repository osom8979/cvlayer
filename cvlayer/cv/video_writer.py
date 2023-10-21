# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final, Optional, Sequence

import cv2
from numpy.typing import NDArray

from cvlayer.cv.fourcc import DEFAULT_FOURCC, get_fourcc
from cvlayer.typing import SizeI

DEFAULT_FPS: Final[float] = 30.0
AUTO_DETECTION_NSTRIPES: Final[int] = -1


@unique
class VideoWriterProperty(Enum):
    QUALITY = cv2.VIDEOWRITER_PROP_QUALITY
    """
    Current quality (0..100%) of the encoded videostream.
    Can be adjusted dynamically in some codecs.
    """

    FRAMEBYTES = cv2.VIDEOWRITER_PROP_FRAMEBYTES
    """
    (Read-only): Size of just encoded video frame.
    Note that the encoding order may be different from representation order.
    """

    NSTRIPES = cv2.VIDEOWRITER_PROP_NSTRIPES
    """Number of stripes for parallel encoding. -1 for auto detection."""


class VideoWriter:
    def __init__(
        self,
        filename: Optional[str] = None,
        size: Optional[SizeI] = None,
        fps=DEFAULT_FPS,
        fourcc=DEFAULT_FOURCC,
        *,
        color=True,
        api: Optional[int] = None,
        params: Optional[Sequence[int]] = None,
    ):
        if filename is not None:
            if size is None:
                raise ValueError("The 'size' argument is required")

            if api is not None:
                if params is not None:
                    self._writer = cv2.VideoWriter(
                        filename=filename,
                        apiPreference=api,
                        fourcc=fourcc,
                        fps=fps,
                        frameSize=size,
                        params=params,
                    )
                else:
                    self._writer = cv2.VideoWriter(
                        filename=filename,
                        apiPreference=api,
                        fourcc=fourcc,
                        fps=fps,
                        frameSize=size,
                        isColor=color,
                    )
            else:
                if params is not None:
                    self._writer = cv2.VideoWriter(
                        filename=filename,
                        fourcc=fourcc,
                        fps=fps,
                        frameSize=size,
                        params=params,
                    )
                else:
                    self._writer = cv2.VideoWriter(
                        filename=filename,
                        fourcc=fourcc,
                        fps=fps,
                        frameSize=size,
                        isColor=color,
                    )
        else:
            self._writer = cv2.VideoWriter()

    @property
    def writer(self) -> cv2.VideoWriter:
        return self._writer

    def open(
        self,
        filename: str,
        size: SizeI,
        fps=DEFAULT_FPS,
        fourcc=DEFAULT_FOURCC,
        *,
        color=True,
        api: Optional[int] = None,
        params: Optional[Sequence[int]] = None,
    ) -> bool:
        if api is not None:
            if params is not None:
                return self._writer.open(
                    filename=filename,
                    apiPreference=api,
                    fourcc=fourcc,
                    fps=fps,
                    frameSize=size,
                    params=params,
                )
            else:
                return self._writer.open(
                    filename=filename,
                    apiPreference=api,
                    fourcc=fourcc,
                    fps=fps,
                    frameSize=size,
                    isColor=color,
                )
        else:
            if params is not None:
                return self._writer.open(
                    filename=filename,
                    fourcc=fourcc,
                    fps=fps,
                    frameSize=size,
                    params=params,
                )
            else:
                return self._writer.open(
                    filename=filename,
                    fourcc=fourcc,
                    fps=fps,
                    frameSize=size,
                    isColor=color,
                )

    @property
    def opened(self) -> bool:
        return self._writer.isOpened()

    def release(self) -> None:
        return self._writer.release()

    def write(self, image: NDArray) -> None:
        return self._writer.write(image)

    def set(self, prop: int, value: float) -> bool:
        return self._writer.set(prop, value)

    def get(self, prop: int) -> float:
        return self._writer.get(prop)

    def set_property(self, prop: VideoWriterProperty, value: float) -> bool:
        return self.set(prop.value, value)

    def get_property(self, prop: VideoWriterProperty) -> float:
        return self.get(prop.value)

    @property
    def quality(self) -> int:
        return int(self.get_property(VideoWriterProperty.QUALITY))

    @quality.setter
    def quality(self, value: int) -> None:
        assert 0 <= value <= 100
        self.set_property(VideoWriterProperty.QUALITY, float(value))

    @property
    def framebytes(self) -> int:
        return int(self.get_property(VideoWriterProperty.FRAMEBYTES))

    @framebytes.setter
    def framebytes(self, value: int) -> None:
        raise RuntimeError("This is a read-only property")

    @property
    def nstripes(self) -> int:
        return int(self.get_property(VideoWriterProperty.NSTRIPES))

    @nstripes.setter
    def nstripes(self, value: int) -> None:
        self.set_property(VideoWriterProperty.NSTRIPES, float(value))

    @property
    def is_auto_detection_nstripes(self) -> bool:
        return self.nstripes == AUTO_DETECTION_NSTRIPES

    def set_auto_detection_nstripes(self) -> None:
        self.nstripes = AUTO_DETECTION_NSTRIPES

    @staticmethod
    def fourcc(c1: str, c2: str, c3: str, c4: str) -> int:
        return get_fourcc(c1, c2, c3, c4)

    def get_backend_name(self) -> str:
        return self._writer.getBackendName()


class CvlVideoWriter:
    @staticmethod
    def cvl_create_video_writer(
        filename: Optional[str] = None,
        size: Optional[SizeI] = None,
        fps=DEFAULT_FPS,
        fourcc=DEFAULT_FOURCC,
        *,
        color=True,
        api: Optional[int] = None,
        params: Optional[Sequence[int]] = None,
    ) -> VideoWriter:
        return VideoWriter(
            filename=filename,
            size=size,
            fps=fps,
            fourcc=fourcc,
            color=color,
            api=api,
            params=params,
        )

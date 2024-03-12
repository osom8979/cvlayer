# -*- coding: utf-8 -*-

from argparse import Namespace
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto, unique
from io import StringIO
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, Logger
from math import isclose
from os import W_OK, access, getcwd, mkdir, path
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Union

from numpy import float32, float64, full, uint8, zeros_like
from numpy.typing import NDArray

from cvlayer.cv.basic import channels_max, channels_mean, channels_min
from cvlayer.cv.color import PIXEL_8BIT_MAX
from cvlayer.cv.cvt_color import cvt_color
from cvlayer.cv.drawable.defaults import DEFAULT_FONT_FACE
from cvlayer.cv.drawable.rectangle import draw_rectangle
from cvlayer.cv.drawable.text.multiline.box import draw_multiline_text_box
from cvlayer.cv.fourcc import FOURCC_MP4V
from cvlayer.cv.histogram import PADDING as HISTOGRAM_PADDING
from cvlayer.cv.histogram import draw_histogram_channels_with_decorate
from cvlayer.cv.image_io import image_write
from cvlayer.cv.image_resize import resize_ratio
from cvlayer.cv.keymap import (
    KEYCODE_NULL,
    KEYCODE_TIMEOUT,
    HighGuiKeyCode,
    has_highgui_arrow_keys,
    highgui_keys,
)
from cvlayer.cv.mouse import EventFlags, MouseEvent
from cvlayer.cv.roi import normalize_image_roi
from cvlayer.cv.types.color import Color, ColorLike, normalize_color
from cvlayer.cv.types.cvt_color_code import CvtColorCode
from cvlayer.cv.types.interpolation import DEFAULT_INTERPOLATION
from cvlayer.cv.video_capture import VideoCapture
from cvlayer.cv.video_writer import VideoWriter
from cvlayer.cv.window import WINDOW_NORMAL, Window
from cvlayer.debug.avg_stat import AvgStat
from cvlayer.inspect.member import get_public_instance_attributes
from cvlayer.keymap.create import create_callable_keymap
from cvlayer.layer.base import LayerBase
from cvlayer.layer.manager.cvmanager import CvManager
from cvlayer.layer.manager.interface import LayerManagerInterface
from cvlayer.palette.basic import GREEN, RED, WHITE, YELLOW
from cvlayer.palette.flat import CLOUDS_50, MIDNIGHT_BLUE_900
from cvlayer.typing import PointF, PointI, RectI, SizeI, override

DEFAULT_WINDOW_EX_TITLE: Final[str] = "CvWindow"
DEFAULT_LOGGER_NAME: Final[str] = "cvlayer.cvwindow"
DEFAULT_HELP_OFFSET: Final[PointI] = 0, 0
DEFAULT_HELP_ANCHOR: Final[PointF] = 0.0, 0.0
DEFAULT_PLOT_SIZE: Final[SizeI] = 256, 256
DEFAULT_ROI_COLOR: Final[Color] = RED
DEFAULT_ROI_THICKNESS: Final[int] = 1
DEFAULT_TOAST_ANCHOR: Final[PointF] = 1.0, 1.0
DEFAULT_TOAST_COLOR: Final[Color] = WHITE
DEFAULT_TOAST_DURATION: Final[float] = 2.0


@unique
class HelpMode(Enum):
    HIDE = auto()
    INFO = auto()
    DEBUG = auto()


@dataclass
class KeyDefine:
    quit: Sequence[str] = field(default_factory=list)
    play: Sequence[str] = field(default_factory=list)
    help: Sequence[str] = field(default_factory=list)
    manpage: Sequence[str] = field(default_factory=list)
    snapshot: Sequence[str] = field(default_factory=list)
    wait_down: Sequence[str] = field(default_factory=list)
    wait_up: Sequence[str] = field(default_factory=list)
    font_down: Sequence[str] = field(default_factory=list)
    font_up: Sequence[str] = field(default_factory=list)
    layer_select: Sequence[str] = field(default_factory=list)
    layer_prev: Sequence[str] = field(default_factory=list)
    layer_next: Sequence[str] = field(default_factory=list)
    layer_last: Sequence[str] = field(default_factory=list)
    select_roi: Sequence[str] = field(default_factory=list)
    frame_prev: Sequence[str] = field(default_factory=list)
    frame_next: Sequence[str] = field(default_factory=list)
    frame_begin: Sequence[str] = field(default_factory=list)
    frame_end: Sequence[str] = field(default_factory=list)
    param_prev: Sequence[str] = field(default_factory=list)
    param_next: Sequence[str] = field(default_factory=list)
    param_down: Sequence[str] = field(default_factory=list)
    param_up: Sequence[str] = field(default_factory=list)

    @classmethod
    def defaults(cls):
        return cls(
            quit=["Q", "q"],
            play=[" "],
            help=["H", "h"],
            manpage=["/", "?"],
            snapshot=["`", "\n"],
            wait_down=["<", ","],
            wait_up=[">", "."],
            font_down=["-", "_"],
            font_up=["=", "+"],
            layer_select=[str(i) for i in range(10)],
            layer_prev=["{", "["],
            layer_next=["}", "]"],
            layer_last=["\\", "|"],
            select_roi=["R", "r"],
            frame_prev=["P", "p"],
            frame_next=["N", "n"],
            frame_begin=["B", "b"],
            frame_end=["E", "e"],
            param_prev=["W", "w"],
            param_next=["S", "s"],
            param_down=["A", "a"],
            param_up=["D", "d"],
        )


@dataclass
class FrameEventCallable:
    handler: Callable[[], None]
    counter: int

    def __call__(self) -> None:
        if self.counter >= 1:
            self.counter -= 1
            self.handler()
        elif self.counter <= -1:
            self.handler()
        else:
            assert self.counter == 0


def analyze_frame_as_text(frame: NDArray, roi: Optional[RectI] = None) -> str:
    if roi is not None:
        x1, y1, x2, y2 = normalize_image_roi(frame, roi)
        if (x2 - x1) * (y2 - y1) != 0:
            frame = frame[y1:y2, x1:x2]

    buffer = StringIO()
    buffer.write(f"Shape: {list(frame.shape)}\n")
    buffer.write(f"Means: {[round(m) for m in channels_mean(frame)]}\n")
    buffer.write(f"Min: {[round(m) for m in channels_min(frame)]}\n")
    buffer.write(f"Max: {[round(m) for m in channels_max(frame)]}")
    return buffer.getvalue()


class CvWindow(LayerManagerInterface, Window):
    _writer: Optional[VideoWriter]
    _frame_events: Dict[int, List[FrameEventCallable]]

    def __init__(
        self,
        input: str,  # noqa
        output: Optional[str] = None,
        font=DEFAULT_FONT_FACE,
        font_scale=1.0,
        preview_scale=1.0,
        preview_scale_method=DEFAULT_INTERPOLATION,
        start_position=0,
        help_mode=HelpMode.DEBUG,
        play=False,
        headless=False,
        show_manual=False,
        show_toast=True,
        verbose=0,
        keymap: Optional[KeyDefine] = None,
        window_title=DEFAULT_WINDOW_EX_TITLE,
        window_flags=WINDOW_NORMAL,
        window_wait=1,
        window_size: Optional[SizeI] = None,
        window_position: Optional[PointI] = None,
        writer_size: Optional[SizeI] = None,
        writer_fps: Optional[float] = None,
        writer_fourcc=FOURCC_MP4V,
        logger: Optional[Union[Logger, str]] = DEFAULT_LOGGER_NAME,
        logging_step=1,
        snapshot_base: Optional[str] = None,
        snapshot_ext: Optional[str] = None,
        help_offset: Optional[PointI] = None,
        help_anchor: Optional[PointF] = None,
        plot_size: Optional[SizeI] = None,
        plot_padding=HISTOGRAM_PADDING,
        roi: Optional[RectI] = None,
        roi_color=DEFAULT_ROI_COLOR,
        roi_thickness=DEFAULT_ROI_THICKNESS,
        roi_draw=True,
        toast_anchor: Optional[PointF] = None,
        toast_color=DEFAULT_TOAST_COLOR,
        toast_duration=DEFAULT_TOAST_DURATION,
        manager: Optional[CvManager] = None,
        use_deepcopy=False,
    ):
        super().__init__(window_title, window_flags, suppress_init=headless)

        assert 0 <= start_position
        assert 1 <= window_wait

        self._input = input
        self._output = output
        self._font = font
        self._font_scale = font_scale
        self._font_scale_step = 0.05
        self._preview_scale = preview_scale
        self._preview_scale_method = preview_scale_method
        self._window_wait = window_wait
        self._help_mode = help_mode
        self._play = play
        self._headless = headless
        self._show_manual = show_manual
        self._show_toast = show_toast
        self._verbose = verbose
        self._snapshot_base = snapshot_base if snapshot_base else getcwd()
        self._snapshot_ext = snapshot_ext if snapshot_ext else ".png"
        self._help_offset = help_offset if help_offset else DEFAULT_HELP_OFFSET
        self._help_anchor = help_anchor if help_anchor else DEFAULT_HELP_ANCHOR
        self._plot_size = plot_size if plot_size else DEFAULT_PLOT_SIZE
        self._plot_padding = plot_padding
        self._roi_color = roi_color
        self._roi_thickness = roi_thickness
        self._roi_draw = roi_draw
        self._toast_anchor = toast_anchor if toast_anchor else DEFAULT_TOAST_ANCHOR
        self._toast_duration = toast_duration
        self._toast_text = str()
        self._toast_color = toast_color
        self._toast_begin = datetime.now()
        self._use_deepcopy = use_deepcopy

        self._manager = manager if manager else CvManager(logger=logger)
        self._manager.set_roi(roi)

        if not self._headless and window_size is not None:
            win_width, win_height = window_size
            if win_width <= 0 or win_height <= 0:
                raise ValueError(f"Invalid window size: {window_size}")
            self.resize(win_width, win_height)

        if not self._headless and window_position is not None:
            win_x, win_y = window_position
            if win_x <= 0 or win_y <= 0:
                raise ValueError(f"Invalid window position: {window_position}")
            self.move(win_x, win_y)

        self._capture = VideoCapture(self._input)
        if not self._capture.opened:
            raise RuntimeError("A Video Capture was created but not opened")
        if self._capture.width < 1:
            raise RuntimeError("Invalid input video's width")
        if self._capture.height < 1:
            raise RuntimeError("Invalid input video's height")
        if self._capture.fps < 1:
            raise RuntimeError("Invalid input video's FPS")

        # ------------------------------------------------------------------------------
        # Don't check the number of frames.
        # In some cases, the number of frames is unknown. (e.g. network stream)
        # if self._capture.frames < 1:
        #     raise RuntimeError("Invalid input video's frame count")
        # ------------------------------------------------------------------------------

        width = self._capture.width
        height = self._capture.height
        self._capture.pos = start_position

        retval, frame = self._capture.read()
        if not retval:
            raise EOFError("Failed to read the first frame")

        self._empty_frame = zeros_like(frame, dtype=uint8)
        self._original_frame = frame.copy()
        self._preview_frame = frame.copy()

        self._select_roi_mode = False
        self._select_roi_button_down = False

        if self._output:
            size = writer_size if writer_size is not None else (width, height)
            fps = writer_fps if writer_fps is not None else self._capture.fps
            self._writer = VideoWriter(self._output, size, fps, writer_fourcc)
            if not self._writer.opened:
                raise RuntimeError("A Video Writer was created but not opened")
        else:
            self._writer = None

        keymap = keymap if keymap else KeyDefine.defaults()
        assert keymap is not None
        keymap_attrs = get_public_instance_attributes(keymap)

        shortcut = dict()
        buffer = StringIO()
        buffer.write("Keyboard shortcuts\n")
        for name, keys in keymap_attrs:
            assert isinstance(keys, list)
            shortcut[name] = [ord(k) for k in keys]
            buffer.write(f" {name}: {keys}\n")

        self._keymap_manual_text = buffer.getvalue()
        self._keymap = create_callable_keymap(self, shortcut)

        self._highgui_keys = highgui_keys()
        self._has_arrow_keys = has_highgui_arrow_keys(self._highgui_keys)

        self._mouse_event = MouseEvent.MOUSE_MOVE
        self._mouse_x = 0
        self._mouse_y = 0
        self._mouse_flags = 0
        self._keycode = 0

        self._stat = AvgStat("Iter", self._manager.logger, logging_step, verbose, 1)
        self._process_duration = 0.0
        self._shutdown = False

        self._frame_events = dict()

    @staticmethod
    def namespace_to_dict(ns: Namespace) -> Dict[str, Any]:
        return {k: v for k, v in get_public_instance_attributes(ns)}

    @classmethod
    def from_namespace(cls, ns: Namespace):
        return cls(**cls.namespace_to_dict(ns))

    @override
    def layer(self, key: Any) -> LayerBase:
        return self._manager.layer(key)

    @override
    def set_roi(self, roi: Any) -> None:
        self._manager.set_roi(roi)

    @property
    def frames(self) -> int:
        return self._capture.frames

    @property
    def pos(self) -> int:
        return self._capture.pos

    @property
    def logger(self):
        return self._manager.logger

    @property
    def roi(self):
        return self._manager.roi

    @roi.setter
    def roi(self, value: Optional[RectI]) -> None:
        self._manager.set_roi(value)

    @property
    def original_frame(self) -> NDArray:
        return self._original_frame

    @property
    def preview_frame(self) -> NDArray:
        return self._preview_frame

    @property
    def last_frame(self) -> NDArray:
        return self._manager.last_layer.frame

    @property
    def last_data(self) -> Any:
        return self._manager.last_layer.data

    @property
    def mouse_event(self):
        return self._mouse_event

    @property
    def mouse_x(self):
        return self._mouse_x

    @property
    def mouse_y(self):
        return self._mouse_y

    @property
    def mouse_flags(self):
        return self._mouse_flags

    @property
    def keycode(self):
        return self._keycode

    def add_frame_event(self, index: int, event: Callable[[], None], counter=1) -> None:
        assert index >= 0
        if index in self._frame_events:
            self._frame_events[index].append(FrameEventCallable(event, counter))
        else:
            self._frame_events[index] = [FrameEventCallable(event, counter)]

    def clear_toast(self) -> None:
        self._toast_text = str()

    def toast(
        self,
        text: str,
        color: Optional[ColorLike] = None,
        duration: Optional[float] = None,
        level: Optional[int] = None,
    ) -> None:
        self._toast_text = text
        if color is not None:
            self._toast_color = normalize_color(color)
        self._toast_begin = datetime.now()
        if duration is not None:
            diff = duration - self._toast_duration
            self._toast_begin -= timedelta(seconds=diff)
        if level is not None:
            self._manager.logger.log(level, text)

    def toast_debug(self, text: str, duration: Optional[float] = None) -> None:
        self.toast(text, color=GREEN, duration=duration, level=DEBUG)

    def toast_info(self, text: str, duration: Optional[float] = None) -> None:
        self.toast(text, color=WHITE, duration=duration, level=INFO)

    def toast_warning(self, text: str, duration: Optional[float] = None) -> None:
        self.toast(text, color=YELLOW, duration=duration, level=WARNING)

    def toast_error(self, text: str, duration: Optional[float] = None) -> None:
        self.toast(text, color=RED, duration=duration, level=ERROR)

    def toast_critical(self, text: str, duration: Optional[float] = None) -> None:
        self.toast(text, color=RED, duration=duration, level=CRITICAL)

    def on_keydown_quit(self, keycode: int) -> None:
        raise KeyboardInterrupt("Quit key detected")

    def on_keydown_play(self, keycode: int) -> None:
        assert 0 < keycode
        self.flip_play()

    def on_keydown_help(self, keycode: int) -> None:
        assert 0 < keycode
        self.flip_help_popup()

    def on_keydown_manpage(self, keycode: int) -> None:
        assert 0 < keycode
        self.flip_manual_page()

    def on_keydown_snapshot(self, keycode: int) -> None:
        assert 0 < keycode
        self.snapshot()

    def on_keydown_wait_down(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_wait_down()

    def on_keydown_wait_up(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_wait_up()

    def on_keydown_font_down(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_font_down()

    def on_keydown_font_up(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_font_up()

    def on_keydown_layer_select(self, keycode: int) -> None:
        assert 0 < keycode
        index = keycode - ord("0")
        assert 0 <= index <= 9
        self.do_layer_select(index)

    def on_keydown_layer_prev(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_layer_prev()

    def on_keydown_layer_next(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_layer_next()

    def on_keydown_layer_last(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_layer_last()

    def on_keydown_select_roi(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_select_roi()

    def on_keydown_frame_prev(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_frame_prev()

    def on_keydown_frame_next(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_frame_next()

    def on_keydown_frame_begin(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_frame_begin()

    def on_keydown_frame_end(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_frame_end()

    def on_keydown_param_prev(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_param_prev()

    def on_keydown_param_next(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_param_next()

    def on_keydown_param_down(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_param_down()

    def on_keydown_param_up(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_param_up()

    def on_create(self) -> None:
        self._manager.on_create()

    def on_destroy(self) -> None:
        self._manager.on_destroy()

        if self._writer is not None:
            assert self._writer.opened
            self._writer.release()

    def on_frame(self, image: NDArray) -> Optional[NDArray]:
        self._manager.run(image, self._use_deepcopy)
        return None

    def on_keydown(self, keycode: int) -> None:
        # Process user events with priority.
        # And if it consumed the event with a return value of `True`,
        # It doesn't handle default events.
        if self._manager.on_keydown(keycode):
            return

        if keycode in self._keymap:
            self._keymap[keycode](keycode)
            return

        if self._has_arrow_keys:
            if keycode == self._highgui_keys[HighGuiKeyCode.ARROW_UP]:
                self.do_param_prev()
            elif keycode == self._highgui_keys[HighGuiKeyCode.ARROW_DOWN]:
                self.do_param_next()
            elif keycode == self._highgui_keys[HighGuiKeyCode.ARROW_LEFT]:
                self.do_param_down()
            elif keycode == self._highgui_keys[HighGuiKeyCode.ARROW_RIGHT]:
                self.do_param_up()

    def on_mouse(self, event: MouseEvent, x: int, y: int, flags: EventFlags) -> None:
        self._mouse_event = event
        self._mouse_x = x
        self._mouse_y = y
        self._mouse_flags = flags

        # Intercept mouse events for select ROI mode.
        if self._select_roi_mode:
            if event == MouseEvent.LBUTTON_DOWN:
                self.roi = x, y, x, y
                self._select_roi_button_down = True
            if self._select_roi_button_down:
                if event == MouseEvent.MOUSE_MOVE:
                    self.roi = self.roi[0], self.roi[1], x, y
                elif event == MouseEvent.LBUTTON_UP:
                    if self.roi[0] == x and self.roi[1] == y:
                        self.set_roi(None)
                    else:
                        self.roi = self.roi[0], self.roi[1], x, y
                    self._select_roi_mode = False
                    self._select_roi_button_down = False
                    self.clear_toast()
            return

        # Process user events with priority.
        # And if it consumed the event with a return value of `True`,
        # It doesn't handle default events.
        if self._manager.on_mouse(event, x, y, flags):
            return

        # TODO: Global mouse event implementation to be added later

    def on_trackbar(self, name: str, value: int) -> None:
        pass

    def shutdown(self) -> None:
        self._shutdown = True

    def read_next_frame(self) -> NDArray:
        retval, frame = self._capture.read()
        if not retval:
            raise EOFError("Failed to read the next frame")
        return frame

    def read_prev_frame(self) -> NDArray:
        self._capture.pos -= 2
        try:
            return self.read_next_frame()
        except EOFError:
            raise EOFError("Failed to read the prev frame")

    def read_first_frame(self) -> NDArray:
        self._capture.pos = 0
        try:
            return self.read_next_frame()
        except EOFError:
            raise EOFError("Failed to read the prev frame")

    def read_last_frame(self) -> NDArray:
        self._capture.pos = self._capture.frames - 1
        try:
            return self.read_next_frame()
        except EOFError:
            raise EOFError("Failed to read the prev frame")

    def flip_play(self) -> None:
        self._play = not self._play
        state_text = "played" if self._play else "stopped"
        self.logger.info(f"The video has been {state_text}")

    def flip_help_popup(self) -> None:
        if self._help_mode == HelpMode.HIDE:
            self._help_mode = HelpMode.INFO
        elif self._help_mode == HelpMode.INFO:
            self._help_mode = HelpMode.DEBUG
        elif self._help_mode == HelpMode.DEBUG:
            self._help_mode = HelpMode.HIDE
        else:
            assert False, "Inaccessible section"
        self.logger.info(f"{self._help_mode.name} help popup")

    def flip_manual_page(self) -> None:
        self._show_manual = not self._show_manual
        popup_state = "Show" if self._show_manual else "Hide"
        self.logger.info(f"{popup_state} man page")

    def snapshot(
        self,
        directory: Optional[str] = None,
        image_extension: Optional[str] = None,
    ) -> None:
        if directory:
            base = directory
        else:
            if self._snapshot_base:
                if path.isdir(self._snapshot_base):
                    base = self._snapshot_base
                else:
                    base = path.dirname(self._snapshot_base)
            else:
                base = getcwd()

        if not path.isdir(base):
            raise NotADirectoryError(f"'{base}' is not a directory")
        if not access(base, W_OK):
            raise PermissionError(f"Write access to directory '{base}' is required")

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = path.join(base, f"{self._capture.pos}-{now}")

        if not path.isdir(prefix):
            mkdir(prefix)
            self.logger.debug(f"Make directory: '{prefix}'")

        self.logger.debug(f"Saving all layer snapshots as '{prefix}' directory ...")
        ext = image_extension if image_extension else self._snapshot_ext

        for index, layer in enumerate(self._manager.values()):
            assert isinstance(layer, LayerBase)
            filename = f"layer{index}-{layer.name}{ext}"
            image_write(path.join(prefix, filename), layer.frame)

        image_write(path.join(prefix, f"original{ext}"), self._original_frame)
        image_write(path.join(prefix, f"preview{ext}"), self._preview_frame)

        self.toast_info(f"Write snapshots: '{prefix}'")

    def do_wait_up(self) -> None:
        self._window_wait += 1
        self.toast_info(f"Increase wait milliseconds to {self._window_wait}ms")

    def do_wait_down(self) -> None:
        if self._window_wait >= 2:
            self._window_wait -= 1
        self.toast_info(f"Decrease wait milliseconds to {self._window_wait}ms")

    def do_font_up(self) -> None:
        self._font_scale = round(self._font_scale + self._font_scale_step, 2)
        self.toast_info(f"Increase font scale: {self._font_scale:.2f}")

    def do_font_down(self) -> None:
        self._font_scale = round(self._font_scale - self._font_scale_step, 2)
        if self._font_scale < 0.0:
            self._font_scale = 0.0
        self.toast_info(f"Decrease font scale: {self._font_scale:.2f}")

    def do_layer_select(self, index: int) -> None:
        self._manager.set_cursor(index)
        self.logger.info(self._manager.as_current_layer_info_text())

    def do_layer_prev(self) -> None:
        self._manager.move_prev_layer()
        self.logger.info(self._manager.as_current_layer_info_text())

    def do_layer_next(self) -> None:
        self._manager.move_next_layer()
        self.logger.info(self._manager.as_current_layer_info_text())

    def do_layer_last(self) -> None:
        self._manager.move_last_layer()

    def do_select_roi(self) -> None:
        if self._select_roi_mode:
            self._select_roi_mode = False
            self.set_roi(None)
            self.toast_info("Clear ROI")
        else:
            self._select_roi_mode = True
            self.toast_info("Select ROI ...")

    def do_frame_prev(self) -> None:
        self._original_frame = self.read_prev_frame()

    def do_frame_next(self) -> None:
        self._original_frame = self.read_next_frame()

    def do_frame_begin(self) -> None:
        self._original_frame = self.read_first_frame()

    def do_frame_end(self) -> None:
        self._original_frame = self.read_last_frame()

    def do_param_prev(self) -> None:
        if self._manager.is_cursor_at_last:
            raise IndexError("The current layer is the last layer")

        self._manager.current_layer.prev_cursor()
        self.logger.info(self._manager.as_current_param_info_text())

    def do_param_next(self) -> None:
        if self._manager.is_cursor_at_last:
            raise IndexError("The current layer is the last layer")

        self._manager.current_layer.next_cursor()
        self.logger.info(self._manager.as_current_param_info_text())

    def do_param_up(self) -> None:
        if self._manager.is_cursor_at_last:
            raise IndexError("The current layer is the last layer")

        self._manager.current_layer.increase_at_cursor()
        self.logger.info(self._manager.as_current_param_info_text())

    def do_param_down(self) -> None:
        if self._manager.is_cursor_at_last:
            raise IndexError("The current layer is the last layer")

        self._manager.current_layer.decrease_at_cursor()
        self.logger.info(self._manager.as_current_param_info_text())

    def do_process(self, frame: NDArray) -> Optional[NDArray]:
        begin = datetime.now()
        try:
            if self._use_deepcopy:
                frame = frame.copy()
            self._manager.update_first_frame_and_data(frame)
            return self.on_frame(frame)
        except BaseException as e:
            self.logger.exception(e)
            return None
        finally:
            self._process_duration = (datetime.now() - begin).total_seconds()

    def as_information_text(self) -> str:
        duration = self._stat.avg
        fps = 1.0 / duration if duration != 0 else 0
        cursor = self._manager.cursor
        number_of_layers = self._manager.number_of_layers

        buffer = StringIO()
        buffer.write(f"Frame {self._capture.pos}/{self._capture.frames}\n")
        buffer.write(f"FPS: {fps:.1f} (duration={duration:.3f}s)\n")
        buffer.write(f"Layer index: {cursor}/{number_of_layers}\n")
        buffer.write(f"Process duration: {self._process_duration:.3f}s\n")
        buffer.write(f"Layers total duration: {self._manager.total_duration:.3f}s\n")

        if self._manager.is_cursor_at_last:
            buffer.write("[Last layer]")
        else:
            layer_duration = self._manager.current_layer.duration
            buffer.write(f"Layer duration: {layer_duration:.3f}s\n")
            buffer.write(self._manager.current_layer.as_help())

        return buffer.getvalue()

    def create_manpage(self, size: SizeI) -> NDArray:
        manpage_shape = size[1], size[0], 3
        manpage_frame = full(manpage_shape, fill_value=MIDNIGHT_BLUE_900, dtype=uint8)
        assert len(manpage_frame.shape) == 3
        assert manpage_frame.shape[2] == 3
        draw_multiline_text_box(
            manpage_frame,
            self._keymap_manual_text,
            pos=(0, 0),
            font=self._font,
            scale=self._font_scale,
            color=CLOUDS_50,
            background_alpha=0,
        )
        return manpage_frame

    def _select_preview_source(self, result_frame: Optional[NDArray]) -> NDArray:
        if self._show_manual:
            ref_frame = result_frame if result_frame is not None else self._empty_frame
            manpage_size = ref_frame.shape[0], ref_frame.shape[1]
            return self.create_manpage(manpage_size)

        if self._manager.is_cursor_at_last:
            if result_frame is not None:
                return result_frame

            if self._verbose >= 1:
                self.logger.warning("The result frame does not exist")
            return self._empty_frame

        current_layer = self._manager.current_layer
        if current_layer.frame is not None:
            return current_layer.frame

        if self._verbose >= 1:
            layer_name = current_layer.name
            self.logger.warning(f"The frame of the '{layer_name}' layer does not exist")
            if self._verbose >= 2 and current_layer.has_error:
                self.logger.exception(current_layer.error)

        return self._empty_frame

    def _coloring(self, frame: NDArray) -> NDArray:
        if frame.dtype in (float32, float64):
            assert frame.min() >= 0.0
            assert frame.max() <= 1.0
            assert PIXEL_8BIT_MAX == 255
            frame = (frame * PIXEL_8BIT_MAX).astype(uint8)
        if len(frame.shape) == 2:
            return cvt_color(frame, CvtColorCode.GRAY2BGR)
        else:
            return frame.copy() if self._use_deepcopy else frame

    def _resizing(self, frame: NDArray) -> NDArray:
        if isclose(self._preview_scale, 1.0):
            return frame.copy() if self._use_deepcopy else frame
        else:
            sx = self._preview_scale
            sy = self._preview_scale
            sm = self._preview_scale_method
            return resize_ratio(frame, sx, sy, sm)

    def _draw_histogram(
        self,
        frame: NDArray,
        help_roi: RectI,
        analyze_frame: NDArray,
    ) -> RectI:
        hx1, hy1, hx2, hy2 = help_roi
        hx1 = hx1 if self._help_anchor[0] < 0.5 else hx2 - self._plot_size[0]
        hy1 = hy2 if self._help_anchor[1] < 0.5 else hy1 - self._plot_size[1]
        hx2 = hx1 + self._plot_size[0] + (self._plot_padding * 2)
        hy2 = hy1 + self._plot_size[1] + (self._plot_padding * 2)
        hist_roi = hx1, hy1, hx2, hy2
        draw_histogram_channels_with_decorate(
            frame,
            hist_roi,
            analyze_frame,
            self.roi,
            padding=self._plot_padding,
        )
        return hist_roi

    def _draw_toast(self, frame: NDArray):
        return draw_multiline_text_box(
            frame,
            text=self._toast_text,
            pos=(0, 0),
            font=self._font,
            scale=self._font_scale,
            color=self._toast_color,
            anchor=self._toast_anchor,
        )

    def _draw_information(self, frame: NDArray, analyze_frame: NDArray) -> NDArray:
        # [IMPORTANT] Do not use `self._use_deepcopy` property.
        canvas = frame.copy()

        if self._roi_draw and self.roi is not None:
            draw_rectangle(canvas, self.roi, self._roi_color, self._roi_thickness)

        buffer = StringIO()
        buffer.write(self.as_information_text())
        if self._help_mode == HelpMode.DEBUG:
            buffer.write("\n" + analyze_frame_as_text(analyze_frame, self.roi))

        _, help_roi = draw_multiline_text_box(
            image=canvas,
            text=buffer.getvalue(),
            pos=self._help_offset,
            font=self._font,
            scale=self._font_scale,
            anchor=self._help_anchor,
        )

        if self._help_mode == HelpMode.DEBUG:
            self._draw_histogram(canvas, help_roi, analyze_frame)

        if self._show_toast and self._toast_text:
            toast_duration = (datetime.now() - self._toast_begin).total_seconds()
            if toast_duration <= self._toast_duration:
                self._draw_toast(canvas)

        return canvas

    def _previewing(self, frame: NDArray, analyze_frame: NDArray) -> NDArray:
        if self._show_manual:
            return frame.copy()

        if self._help_mode == HelpMode.HIDE:
            return frame.copy()

        return self._draw_information(frame, analyze_frame)

    def _iter(self) -> None:
        if not self._capture.opened:
            raise EOFError("Input video is not opened")

        if self._play:
            self._original_frame = self.read_next_frame()

        events = self._frame_events.get(self._capture.pos)
        if events is not None:
            for event in events:
                event()

        result_frame = self.do_process(self._original_frame)
        select_frame = self._select_preview_source(result_frame)
        colored_frame = self._coloring(select_frame)
        resized_frame = self._resizing(colored_frame)
        self._preview_frame = self._previewing(resized_frame, select_frame)

        if self._writer is not None:
            assert self._writer.opened
            self._writer.write(self._preview_frame)

        if self._headless:
            return

        self.draw(self._preview_frame)
        self._keycode = self.wait_key_ex(self._window_wait)

        if not self.visible:
            raise InterruptedError("The window is not visible")

        if self._keycode not in (KEYCODE_NULL, KEYCODE_TIMEOUT):
            try:
                self.on_keydown(self._keycode)
            except KeyboardInterrupt:
                raise
            except InterruptedError:
                raise
            except BaseException as e:
                self.logger.exception(e)

    def run(self) -> None:
        self.on_create()
        try:
            while not self._shutdown:
                with self._stat:
                    self._iter()
        except KeyboardInterrupt as e:
            self.logger.warning(e)
        except InterruptedError as e:
            self.logger.warning(e)
        except EOFError as e:
            self.logger.error(e)
        except BaseException as e:
            self.logger.exception(e)
        finally:
            try:
                self.on_destroy()
            except BaseException as e:
                self.logger.exception(e)
            finally:
                if not self._headless:
                    self.destroy()

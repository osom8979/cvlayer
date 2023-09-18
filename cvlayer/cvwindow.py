# -*- coding: utf-8 -*-

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto, unique
from io import StringIO
from math import isclose
from os import W_OK, access, getcwd, mkdir, path
from typing import Any, Final, List, Optional

from numpy import full_like, uint8, zeros_like
from numpy.typing import NDArray

from cvlayer.cv.basic import mean
from cvlayer.cv.cvt_color import CvtColorCode, cvt_color
from cvlayer.cv.drawable import FONT_HERSHEY_SIMPLEX, draw_multiline_text_box
from cvlayer.cv.fourcc import FOURCC_MP4V
from cvlayer.cv.histogram import draw_histogram_channels
from cvlayer.cv.image_io import image_write
from cvlayer.cv.image_resize import Interpolation, resize_ratio
from cvlayer.cv.keymap import (
    KEYCODE_NULL,
    KEYCODE_TIMEOUT,
    HighGuiKeyCode,
    has_highgui_arrow_keys,
    highgui_keys,
)
from cvlayer.cv.mouse import EventFlags, MouseEvent
from cvlayer.cv.video_capture import VideoCapture
from cvlayer.cv.video_writer import VideoWriter
from cvlayer.cv.window import WINDOW_NORMAL, Window
from cvlayer.debug.avg_stat import AvgStat
from cvlayer.inspect.member import get_public_instance_attributes
from cvlayer.keymap.create import create_callable_keymap
from cvlayer.layers.base.layer_base import LayerBase
from cvlayer.layers.base.layer_manager import LayerManager
from cvlayer.palette.flat import CLOUDS_50, MIDNIGHT_BLUE_900
from cvlayer.typing import PointFloat, PointInt, RectInt, SizeInt

DEFAULT_WINDOW_EX_TITLE: Final[str] = "CvWindow"
DEFAULT_HELP_OFFSET: Final[PointInt] = 0, 0
DEFAULT_HELP_ANCHOR: Final[PointFloat] = 0.0, 0.0
DEFAULT_PLOT_SIZE: Final[SizeInt] = 256, 256


@unique
class HelpMode(Enum):
    HIDE = auto()
    INFO = auto()
    PLOT = auto()


@dataclass
class KeyDefine:
    quit: List[str]
    play: List[str]
    help: List[str]
    manpage: List[str]
    snapshot: List[str]
    wait_down: List[str]
    wait_up: List[str]
    layer_select: List[str]
    layer_prev: List[str]
    layer_next: List[str]
    layer_last: List[str]
    frame_prev: List[str]
    frame_next: List[str]
    param_prev: List[str]
    param_next: List[str]
    param_down: List[str]
    param_up: List[str]

    @classmethod
    def defaults(cls):
        return cls(
            quit=["Q", "q"],
            play=[" "],
            help=["H", "h"],
            manpage=["/", "?"],
            snapshot=["`", "\n"],
            wait_down=["-", "_"],
            wait_up=["=", "+"],
            layer_select=[str(i) for i in range(10)],
            layer_prev=["{", "["],
            layer_next=["}", "]"],
            layer_last=["\\", "|"],
            frame_prev=["P", "p", "<", ","],
            frame_next=["N", "n", ">", "."],
            param_prev=["W", "w"],
            param_next=["S", "s"],
            param_down=["A", "a"],
            param_up=["D", "d"],
        )


class CvWindow(Window):
    _writer: Optional[VideoWriter]

    def __init__(
        self,
        input: str,  # noqa
        output: Optional[str] = None,
        font=FONT_HERSHEY_SIMPLEX,
        font_scale=1.0,
        preview_scale=1.0,
        preview_scale_method=Interpolation.INTER_AREA,
        start_position=0,
        help_mode=HelpMode.INFO,
        play=False,
        headless=False,
        show_man=False,
        verbose=0,
        keymap: Optional[KeyDefine] = None,
        window_title=DEFAULT_WINDOW_EX_TITLE,
        window_flags=WINDOW_NORMAL,
        window_wait=1,
        window_size: Optional[SizeInt] = None,
        window_position: Optional[PointInt] = None,
        writer_size: Optional[SizeInt] = None,
        writer_fps: Optional[float] = None,
        writer_fourcc=FOURCC_MP4V,
        logger_name: Optional[str] = None,
        logging_step=1,
        help_offset: Optional[PointInt] = None,
        help_anchor: Optional[PointFloat] = None,
        plot_size: Optional[SizeInt] = None,
        plot_roi: Optional[RectInt] = None,
        use_deepcopy=False,
    ):
        super().__init__(window_title, window_flags)

        assert 0 <= start_position
        assert 1 <= window_wait

        self._input = input
        self._output = output
        self._font = font
        self._font_scale = font_scale
        self._preview_scale = preview_scale
        self._preview_scale_method = preview_scale_method
        self._window_wait = window_wait
        self._help_mode = help_mode
        self._play = play
        self._headless = headless
        self._show_man = show_man
        self._verbose = verbose
        self._help_offset = help_offset if help_offset else DEFAULT_HELP_OFFSET
        self._help_anchor = help_anchor if help_anchor else DEFAULT_HELP_ANCHOR
        self._plot_size = plot_size if plot_size else DEFAULT_PLOT_SIZE
        self._plot_roi = plot_roi
        self._use_deepcopy = use_deepcopy

        self._manager = LayerManager(logger_name=logger_name)

        if window_size is not None:
            win_width, win_height = window_size
            if win_width <= 0 or win_height <= 0:
                raise ValueError(f"Invalid window size: {window_size}")
            self.resize(win_width, win_height)

        if window_position is not None:
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
        if self._capture.frames < 1:
            raise RuntimeError("Invalid input video's frame count")

        width = self._capture.width
        height = self._capture.height
        self._capture.pos = start_position

        retval, frame = self._capture.read()
        if not retval:
            raise EOFError("Failed to read the first frame")

        self._empty_frame = zeros_like(frame, dtype=uint8)
        self._original_frame = frame
        self._preview_frame = frame

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

        self._keymap = create_callable_keymap(self, shortcut)
        self._manpage = full_like(frame, fill_value=MIDNIGHT_BLUE_900, dtype=uint8)
        assert len(self._manpage.shape) == 3
        assert self._manpage.shape[2] == 3
        draw_multiline_text_box(
            self._manpage,
            buffer.getvalue(),
            0,
            0,
            font=font,
            scale=font_scale,
            color=CLOUDS_50,
            background_alpha=0,
        )

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

    @property
    def logger(self):
        return self._manager.logger

    def layer(self, name: str) -> LayerBase:
        return self._manager.__getitem__(name)

    def has_layer(self, layer: Any) -> bool:
        return self._manager.has_layer(layer)

    def get_layer_frame(self, key: Any) -> NDArray:
        return self._manager.get_layer_frame(key)

    def get_layer_data(self, key: Any) -> Any:
        return self._manager.get_layer_data(key)

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

    @property
    def plot_roi(self):
        return self._plot_roi

    @plot_roi.setter
    def plot_roi(self, value: Optional[RectInt]) -> None:
        self._plot_roi = value

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

    def on_keydown_frame_prev(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_frame_prev()

    def on_keydown_frame_next(self, keycode: int) -> None:
        assert 0 < keycode
        self.do_frame_next()

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

    def flip_play(self) -> None:
        self._play = not self._play
        state_text = "played" if self._play else "stopped"
        self.logger.info(f"The video has been {state_text}")

    def flip_help_popup(self) -> None:
        if self._help_mode == HelpMode.HIDE:
            self._help_mode = HelpMode.INFO
        elif self._help_mode == HelpMode.INFO:
            self._help_mode = HelpMode.PLOT
        elif self._help_mode == HelpMode.PLOT:
            self._help_mode = HelpMode.HIDE
        else:
            assert False, "Inaccessible section"
        self.logger.info(f"{self._help_mode.name} help popup")

    def flip_manual_page(self) -> None:
        self._show_man = not self._show_man
        popup_state = "Show" if self._show_man else "Hide"
        self.logger.info(f"{popup_state} man page")

    def snapshot(self, directory: Optional[str] = None, ext=".png") -> None:
        base = directory if directory else (self._output if self._output else getcwd())
        basedir = path.basename(base)

        if not path.isdir(basedir):
            raise NotADirectoryError(f"'{base}' is not a directory")
        if not access(basedir, W_OK):
            raise PermissionError(f"Write access to directory '{base}' is required")

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = path.join(basedir, f"{self._capture.pos}-{now}")

        if not path.isdir(prefix):
            mkdir(prefix)
            self.logger.debug(f"Make directory; '{prefix}'")

        self.logger.debug(f"Saving all layer snapshots as '{prefix}' directory ...")

        for index in range(self._manager.number_of_layers):
            layer_path = path.join(prefix, f"layer{index}-{ext}")
            layer_frame = self._manager.get_layer_frame(index)
            image_write(layer_path, layer_frame)

        image_write(path.join(prefix, f"original{ext}"), self._original_frame)
        image_write(path.join(prefix, f"preview{ext}"), self._preview_frame)

        self.logger.info(f"Snapshot was successfully saved to directory '{prefix}'")

    def do_wait_up(self) -> None:
        self._window_wait += 1
        self.logger.info(f"Increase wait milliseconds to {self._window_wait}ms")

    def do_wait_down(self) -> None:
        if self._window_wait >= 2:
            self._window_wait -= 1
        self.logger.info(f"Decrease wait milliseconds to {self._window_wait}ms")

    def do_layer_select(self, index: int) -> None:
        self._manager.set_cursor(index)
        self._manager.logging_current_layer()

    def do_layer_prev(self) -> None:
        self._manager.move_prev_layer()
        self._manager.logging_current_layer()

    def do_layer_next(self) -> None:
        self._manager.move_next_layer()
        self._manager.logging_current_layer()

    def do_layer_last(self) -> None:
        self._manager.move_last_layer()
        self.logger.info("Change last layer")

    def do_frame_prev(self) -> None:
        self._original_frame = self.read_prev_frame()

    def do_frame_next(self) -> None:
        self._original_frame = self.read_next_frame()

    def do_param_prev(self) -> None:
        if self._manager.is_cursor_at_last:
            raise IndexError("The current layer is the last layer")

        self._manager.current_layer.prev_cursor()
        self.logger.info("Change prev param cursor")
        self._manager.logging_current_param()

    def do_param_next(self) -> None:
        if self._manager.is_cursor_at_last:
            raise IndexError("The current layer is the last layer")

        self._manager.current_layer.next_cursor()
        self.logger.info("Change next param cursor")
        self._manager.logging_current_param()

    def do_param_up(self) -> None:
        if self._manager.is_cursor_at_last:
            raise IndexError("The current layer is the last layer")

        self._manager.current_layer.increase_at_cursor()
        self.logger.info("Increase value at param cursor")
        self._manager.logging_current_param()

    def do_param_down(self) -> None:
        if self._manager.is_cursor_at_last:
            raise IndexError("The current layer is the last layer")

        self._manager.current_layer.decrease_at_cursor()
        self.logger.info("Decrease value at param cursor")
        self._manager.logging_current_param()

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

    def as_information_text(self, analyze_frame: Optional[NDArray] = None) -> str:
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

        if analyze_frame is not None:
            buffer.write(f"\nMeans: {[round(m) for m in mean(analyze_frame)]}")

        return buffer.getvalue()

    def _select_preview_source(self, result_frame: Optional[NDArray]) -> NDArray:
        if self._show_man:
            return self._manpage
        elif self._manager.is_cursor_at_last:
            if result_frame is not None:
                return result_frame
            else:
                return self._empty_frame
        else:
            try:
                return self._manager.current_layer.frame
            except:  # noqa
                return self._empty_frame

    def _coloring(self, frame: NDArray) -> NDArray:
        assert self
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

    def _draw_information(self, frame: NDArray, analyze_frame: NDArray) -> NDArray:
        if self._use_deepcopy:
            frame = frame.copy()

        if self._show_man:
            return frame

        if self._help_mode == HelpMode.HIDE:
            return frame

        help_roi = draw_multiline_text_box(
            image=frame,
            text=self.as_information_text(analyze_frame),
            x=self._help_offset[0],
            y=self._help_offset[1],
            font=self._font,
            scale=self._font_scale,
            anchor_x=self._help_anchor[0],
            anchor_y=self._help_anchor[1],
        )

        if self._help_mode == HelpMode.PLOT:
            hx1, hy1, hx2, hy2 = help_roi
            dx1 = hx1 if self._help_anchor[0] < 0.5 else hx2 - self._plot_size[0]
            dy1 = hy2 if self._help_anchor[1] < 0.5 else hy1 - self._plot_size[1]
            dx2 = dx1 + self._plot_size[0]
            dy2 = dy1 + self._plot_size[1]
            draw_roi = dx1, dy1, dx2, dy2
            draw_histogram_channels(
                frame,
                draw_roi,
                analyze_frame,
                self._plot_roi,
            )

        return frame

    def _iter(self) -> None:
        if not self._capture.opened:
            raise EOFError("Input video is not opened")

        if self._play:
            self._original_frame = self.read_next_frame()

        result_frame = self.do_process(self._original_frame)
        select_frame = self._select_preview_source(result_frame)
        colored_frame = self._coloring(select_frame)
        resized_frame = self._resizing(colored_frame)
        self._preview_frame = self._draw_information(resized_frame, select_frame)

        if self._writer is not None:
            assert self._writer.opened
            self._writer.write(self._preview_frame)

        if not self._headless:
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
            self.on_destroy()

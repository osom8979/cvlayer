# -*- coding: utf-8 -*-

from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from io import StringIO
from logging import getLogger
from math import isclose
from os import W_OK, access, getcwd, mkdir, path
from typing import Final, List, Optional

from numpy import uint8, zeros_like
from numpy.typing import NDArray
from overrides import override

from cvlayer.cv.cvt_color import CvtColorCode, cvt_color
from cvlayer.cv.drawable import FONT_HERSHEY_SIMPLEX, draw_multiline_text_box
from cvlayer.cv.fourcc import FOURCC_MP4V
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
from cvlayer.typing import PointFloat, PointInt, SizeInt

DEFAULT_WINDOW_EX_TITLE: Final[str] = "LayersWindow"


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
            snapshot=["P", "p"],
            wait_down=["-", "_"],
            wait_up=["=", "+"],
            layer_select=[str(i) for i in range(10)],
            layer_prev=["{", "["],
            layer_next=["}", "]"],
            layer_last=["\\", "|"],
            frame_prev=["<", ","],
            frame_next=[">", "."],
            param_prev=["W", "w"],
            param_next=["S", "s"],
            param_down=["A", "a"],
            param_up=["D", "D"],
        )


class LayersWindow(Window):
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
        play=False,
        headless=False,
        show_help=True,
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
        logging_step=1000,
        help_offset: Optional[PointInt] = None,
        help_anchor: Optional[PointFloat] = None,
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
        self._play = play
        self._headless = headless
        self._show_help = show_help
        self._show_man = show_man
        self._verbose = verbose
        self._help_offset = help_offset if help_offset is not None else (0, 0)
        self._help_anchor = help_anchor if help_anchor is not None else (0.5, 0.0)
        self._use_deepcopy = use_deepcopy

        self._logger = getLogger(logger_name)
        self._manager = LayerManager(name=window_title, logger_name=logger_name)

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

        self._original_frame = frame
        self._processed_frame = frame
        self._resized_preview = frame
        self._record_frame = frame

        if self._output:
            size = writer_size if writer_size is not None else (width, height)
            fps = writer_fps if writer_fps is not None else self._capture.fps
            self._writer = VideoWriter(self._output, size, fps, writer_fourcc)
            if not self._writer.opened:
                raise RuntimeError("A Video Writer was created but not opened")
            self._writer.write(self._original_frame)
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
        self._manpage = zeros_like(frame, dtype=uint8)
        draw_multiline_text_box(self._manpage, buffer.getvalue(), 0, 0)

        self._highgui_keys = highgui_keys()
        self._has_arrow_keys = has_highgui_arrow_keys(self._highgui_keys)

        self._mouse_event = MouseEvent.MOUSE_MOVE
        self._mouse_x = 0
        self._mouse_y = 0
        self._mouse_flags = 0
        self._keycode = 0

        self._stat = AvgStat("Iter", self._logger, logging_step, verbose, 1)
        self._shutdown = False

    def __getitem__(self, item: str):
        if not self._manager.has_layer_by_name(item):
            self._manager.append_layer(LayerBase(item))
        return self._manager.get_layer_by_name(item)

    def __setitem__(self, key: str, value: LayerBase):
        self._manager.append_layer(value)

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
    def logger(self):
        return self._logger

    @property
    def is_last_layer(self):
        return self._manager.is_last_layer

    @property
    def number_of_layers(self):
        return self._manager.number_of_layers

    @property
    def layers(self):
        return self._manager.layers

    @property
    def current_layer_index(self):
        return self._manager.index

    @current_layer_index.setter
    def current_layer_index(self, index: int) -> None:
        self.layer_select(index)

    @property
    def current_layer(self):
        return self._manager.current_layer

    @property
    def process_layers_duration(self) -> float:
        if self.number_of_layers >= 1:
            durations = map(lambda x: x.duration, self.layers)
            return float(reduce(lambda x, y: x + y, durations))
        else:
            return 0.0

    def on_keydown_quit(self, keycode: int) -> None:
        raise InterruptedError("Quit key detected")

    def on_keydown_play(self, keycode: int) -> None:
        assert 0 < keycode
        self.flip_play()

    def on_keydown_help(self, keycode: int) -> None:
        assert 0 < keycode
        self.flip_help()

    def on_keydown_manpage(self, keycode: int) -> None:
        assert 0 < keycode
        self.flip_man()

    def on_keydown_snapshot(self, keycode: int) -> None:
        assert 0 < keycode
        self.snapshot()

    def on_keydown_wait_down(self, keycode: int) -> None:
        assert 0 < keycode
        self.wait_down()

    def on_keydown_wait_up(self, keycode: int) -> None:
        assert 0 < keycode
        self.wait_up()

    def on_keydown_layer_select(self, keycode: int) -> None:
        assert 0 < keycode
        index = keycode - ord("0")
        assert 0 <= index <= 9
        self.layer_select(index)

    def on_keydown_layer_prev(self, keycode: int) -> None:
        assert 0 < keycode
        self.layer_prev()

    def on_keydown_layer_next(self, keycode: int) -> None:
        assert 0 < keycode
        self.layer_next()

    def on_keydown_layer_last(self, keycode: int) -> None:
        assert 0 < keycode
        self.layer_last()

    def on_keydown_frame_prev(self, keycode: int) -> None:
        assert 0 < keycode
        self.frame_prev()

    def on_keydown_frame_next(self, keycode: int) -> None:
        assert 0 < keycode
        self.frame_next()

    def on_keydown_param_prev(self, keycode: int) -> None:
        assert 0 < keycode
        self.param_prev()

    def on_keydown_param_next(self, keycode: int) -> None:
        assert 0 < keycode
        self.param_next()

    def on_keydown_param_down(self, keycode: int) -> None:
        assert 0 < keycode
        self.param_down()

    def on_keydown_param_up(self, keycode: int) -> None:
        assert 0 < keycode
        self.param_up()

    def on_create(self) -> None:
        self._manager.on_create()

    def on_destroy(self) -> None:
        self._manager.on_destroy()

    def on_frame(self, image: NDArray) -> NDArray:
        with self._stat:
            return self._manager.run(image, self._use_deepcopy)[0]

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
                self.param_prev()
            elif keycode == self._highgui_keys[HighGuiKeyCode.ARROW_DOWN]:
                self.param_next()
            elif keycode == self._highgui_keys[HighGuiKeyCode.ARROW_LEFT]:
                self.param_down()
            elif keycode == self._highgui_keys[HighGuiKeyCode.ARROW_RIGHT]:
                self.param_up()

    @override
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

    @override
    def on_trackbar(self, name: str, value: int) -> None:
        pass

    @property
    def preview_frame(self) -> NDArray:
        if self._show_man:
            return self._manpage

        if self.is_last_layer:
            frame = self._resized_preview.copy()
        else:
            frame = self.current_layer.frame.copy()

        if not self._show_help:
            return frame

        buffer = StringIO()
        buffer.write(f"Frame {self._capture.pos}/{self._capture.frames}\n")
        buffer.write(f"Layer {self.current_layer_index}/{self.number_of_layers}\n")

        if self.is_last_layer:
            buffer.write(f"Duration: {self._process_duration:.3f}s\n")
            buffer.write(f"Layers duration: {self.process_layers_duration:.3f}s\n")
            buffer.write("Last layer")
        else:
            buffer.write(f"Duration: {self.current_layer.duration:.3f}s\n")
            buffer.write(self.current_layer.as_help())

        x, y = self._help_offset
        anchor_x, anchor_y = self._help_anchor
        font = self._font
        scale = self._font_scale
        help_text = buffer.getvalue()

        draw_multiline_text_box(
            image=frame,
            text=help_text,
            x=x,
            y=y,
            font=font,
            scale=scale,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )

        return frame

    def shutdown(self) -> None:
        self._shutdown = True

    def read_next_frame(self) -> NDArray:
        retval, frame = self._capture.read()
        if not retval:
            raise EOFError("Failed to read the next frame")
        self._original_frame = frame
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

    def flip_help(self) -> None:
        self._show_help = not self._show_help
        popup_state = "Show" if self._show_help else "Hide"
        self.logger.info(f"{popup_state} help popup")

    def flip_man(self) -> None:
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

        for layer_index in range(self.number_of_layers):
            layer_path = path.join(prefix, f"layer{layer_index}-{ext}")
            image_write(layer_path, self._manager.get_layer_frame(layer_index))

        image_write(path.join(prefix, f"original{ext}"), self._original_frame)
        image_write(path.join(prefix, f"processed{ext}"), self._processed_frame)
        image_write(path.join(prefix, f"resized{ext}"), self._resized_preview)
        image_write(path.join(prefix, f"record{ext}"), self._record_frame)
        image_write(path.join(prefix, f"preview{ext}"), self.preview_frame)

        self.logger.info(f"Snapshot was successfully saved to directory '{prefix}'")

    def wait_up(self) -> None:
        self._window_wait += 1
        self.logger.info(f"Increase wait milliseconds to {self._window_wait}ms")

    def wait_down(self) -> None:
        if self._window_wait >= 2:
            self._window_wait -= 1
        self.logger.info(f"Decrease wait milliseconds to {self._window_wait}ms")

    def layer_select(self, index: int) -> None:
        self._manager.set_index(index)
        self._manager.logging_current_layer()

    def layer_next(self) -> None:
        self._manager.next_layer()
        self._manager.logging_current_layer()

    def layer_prev(self) -> None:
        self._manager.prev_layer()
        self._manager.logging_current_layer()

    def layer_last(self) -> None:
        self._manager.set_last_index()
        self.logger.info("Change last layer")

    def frame_next(self) -> None:
        self.read_next_frame()

    def frame_prev(self) -> None:
        self.read_prev_frame()

    def param_prev(self) -> None:
        if self.is_last_layer:
            raise IndexError("The current layer is the last layer")

        self.current_layer.prev_cursor()
        self.logger.info("Change prev param cursor")
        self._manager.logging_current_param()

    def param_next(self) -> None:
        if self.is_last_layer:
            raise IndexError("The current layer is the last layer")

        self.current_layer.next_cursor()
        self.logger.info("Change next param cursor")
        self._manager.logging_current_param()

    def param_up(self) -> None:
        if self.is_last_layer:
            raise IndexError("The current layer is the last layer")

        self.current_layer.increase_at_cursor()
        self.logger.info("Increase value at param cursor")
        self._manager.logging_current_param()

    def param_down(self) -> None:
        if self.is_last_layer:
            raise IndexError("The current layer is the last layer")

        self.current_layer.decrease_at_cursor()
        self.logger.info("Decrease value at param cursor")
        self._manager.logging_current_param()

    def _iter(self) -> None:
        if not self._capture.opened:
            raise EOFError("Input video is not opened")

        if self._play:
            self.read_next_frame()

        begin = datetime.now()
        try:
            self._processed_frame = self.on_frame(self._original_frame)
        finally:
            self._process_duration = (datetime.now() - begin).total_seconds()

        if isclose(self._preview_scale, 1.0):
            self._resized_preview = self._processed_frame
        else:
            self._resized_preview = resize_ratio(
                self._processed_frame,
                self._preview_scale,
                self._preview_scale,
                self._preview_scale_method,
            )

        if len(self._processed_frame.shape) == 2:
            self._record_frame = cvt_color(self._processed_frame, CvtColorCode.GRAY2BGR)
        else:
            self._record_frame = self._processed_frame

        if self._writer is not None:
            assert self._writer.opened
            self._writer.write(self._record_frame)

        if not self._headless:
            self.draw(self.preview_frame)

        self._keycode = self.wait_key_ex(self._window_wait)

        if not self.visible:
            raise InterruptedError("The window is not visible")

        if self._keycode not in (KEYCODE_NULL, KEYCODE_TIMEOUT):
            self.on_keydown(self._keycode)

    def run(self) -> None:
        self.on_create()
        try:
            while not self._shutdown:
                self._iter()
        except EOFError as e:
            self._logger.error(e)
        except BaseException as e:
            self._logger.exception(e)
        finally:
            self.on_destroy()


class CvlLayersWindow:
    @staticmethod
    def cvl_create_layers_window(
        input: str,  # noqa
        output: Optional[str] = None,
        font=FONT_HERSHEY_SIMPLEX,
        font_scale=1.0,
        preview_scale=1.0,
        preview_scale_method=Interpolation.INTER_AREA,
        start_position=0,
        play=False,
        headless=False,
        show_help=True,
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
        logging_step=1000,
        help_offset: Optional[PointInt] = None,
        help_anchor: Optional[PointFloat] = None,
        use_deepcopy=False,
    ) -> LayersWindow:
        return LayersWindow(
            input=input,
            output=output,
            font=font,
            font_scale=font_scale,
            preview_scale=preview_scale,
            preview_scale_method=preview_scale_method,
            start_position=start_position,
            play=play,
            headless=headless,
            show_help=show_help,
            show_man=show_man,
            verbose=verbose,
            keymap=keymap,
            window_title=window_title,
            window_flags=window_flags,
            window_wait=window_wait,
            window_size=window_size,
            window_position=window_position,
            writer_size=writer_size,
            writer_fps=writer_fps,
            writer_fourcc=writer_fourcc,
            logger_name=logger_name,
            logging_step=logging_step,
            help_offset=help_offset,
            help_anchor=help_anchor,
            use_deepcopy=use_deepcopy,
        )

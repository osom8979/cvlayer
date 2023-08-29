# -*- coding: utf-8 -*-

import os
from datetime import datetime
from functools import lru_cache, reduce
from io import StringIO
from math import floor, isclose
from typing import Any, Dict, List, Optional, Tuple

import cv2
from numpy.typing import NDArray
from overrides import overrides

from cvlayer.cv.drawable import draw_multiline_text
from cvlayer.cv.fourcc import FOURCC_MP4V
from cvlayer.cv.keymap import (
    KEYCODE_ENTER,
    KEYCODE_ESC,
    KEYCODE_NULL,
    HighGuiKeyCode,
    has_highgui_arrow_keys,
    highgui_keys,
)
from cvlayer.cv.mouse import EventFlags, MouseEvent
from cvlayer.keymap.create import (
    DEFAULT_CALLBACK_NAME_PREFIX,
    DEFAULT_CALLBACK_NAME_SUFFIX,
    create_callable_keymap,
)
from cvlayer.layers.base.layer_manager import LayerManager
from cvlayer.np.image import make_image_empty
from cvlayer.vio.vio_interface import VioInterface


@lru_cache
def get_default_keymap():
    return {
        "quit": [ord("Q"), ord("q"), KEYCODE_ESC],
        "play": [ord(" ")],
        "help": [ord("H"), ord("h"), ord("/"), ord("?")],
        "snapshot": [ord("S"), ord("s"), KEYCODE_ENTER],
        "wait_up": [ord("W")],
        "wait_down": [ord("w")],
        "layer_select": [ord(str(i)) for i in range(10)],
        "layer_next": [ord("]"), ord("}")],
        "layer_prev": [ord("["), ord("{")],
        "layer_last": [ord("\\"), ord("|")],
        "frame_next": [ord("N"), ord("n")],
        "frame_prev": [ord("P"), ord("p")],
        "param_next": [ord(">"), ord(".")],
        "param_prev": [ord("<"), ord(",")],
        "param_up": [ord("="), ord("+")],
        "param_down": [ord("-"), ord("_")],
    }


class Vio(LayerManager, VioInterface):
    _capture: Optional[cv2.VideoCapture]
    _writer: Optional[cv2.VideoWriter]

    _process_durations: List[float]
    _layers_durations: List[float]

    def __init__(
        self,
        source: str,
        output: Optional[str] = None,
        preview_scale=1.0,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1.0,
        preview_scale_method=cv2.INTER_AREA,
        start_frame_position=0,
        initial_stop=False,
        headless=False,
        hide_help_popup=False,
        window_title: Optional[str] = None,
        keymap: Optional[Dict[str, List[int]]] = None,
        keymap_callback_prefix=DEFAULT_CALLBACK_NAME_PREFIX,
        keymap_callback_suffix=DEFAULT_CALLBACK_NAME_SUFFIX,
    ):
        super().__init__()

        self._source = source
        self._output = output
        self._preview_scale = preview_scale
        self._font = font
        self._font_scale = font_scale
        self._preview_scale_method = preview_scale_method
        self._start_frame_position = start_frame_position
        self._play = not initial_stop
        self._preview = not headless
        self._help = not hide_help_popup

        self._shortcut_map = create_callable_keymap(
            self,
            keymap if keymap else get_default_keymap(),
            keymap_callback_prefix,
            keymap_callback_suffix,
        )

        self._highgui_keys = highgui_keys()
        self._has_arrow_keys = has_highgui_arrow_keys(self._highgui_keys)

        self._capture = None
        self._writer = None

        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._frames = 0
        self._fourcc = FOURCC_MP4V

        self._title = window_title if window_title else type(self).__name__
        self._frame_index = self._start_frame_position
        self._wait_milliseconds = 1
        self._debugging_step = 10
        self._shutdown = False
        self._help_x = 0
        self._help_y = 0
        self._help_last_layer = "LAST LAYER"
        self._process_duration = 0.0

        self._empty_frame = make_image_empty(800, 600)
        self._original_frame = self._empty_frame
        self._processed_frame = self._empty_frame
        self._resized_preview = self._empty_frame
        self._record_frame = self._empty_frame

        self._process_durations = list()
        self._layers_durations = list()

    @property
    def played(self) -> bool:
        return self._play

    @property
    def paused(self) -> bool:
        return not self._play

    @property
    def show_help(self) -> bool:
        return self._help

    @property
    def process_duration(self) -> float:
        return self._process_duration

    @property
    def process_layers_duration(self) -> float:
        if self.layers:
            durations = map(lambda x: x.duration, self.layers)
            return float(reduce(lambda x, y: x + y, durations))
        else:
            return 0.0

    @property
    def preview_frame(self) -> NDArray:
        if self.is_last_layer:
            frame = self._resized_preview.copy()
        else:
            frame = self.current_layer.next_frame.copy()

        if not self._help:
            return frame

        x = self._help_x
        y = self._help_y
        font = self._font
        scale = self._font_scale
        frame_index = f"Frame {self._frame_index}/{self._frames}\n"
        layer_index = f"Layer [{self.layer_index}/{self.length_layers}] "

        if self.is_last_layer:
            duration = f"Duration: {self._process_duration:.3f}s\n"
            duration += f"Layers duration: {self.process_layers_duration:.3f}s\n"
            text = frame_index + duration + layer_index + self._help_last_layer
        else:
            duration = f"Duration: {self.current_layer.duration:.3f}s\n"
            text = frame_index + duration + layer_index + self.current_layer.to_help()

        draw_multiline_text(frame, text, x, y, font, scale)

        return frame

    def shutdown(self) -> None:
        self._shutdown = True

    def increase_wait_milliseconds(self) -> None:
        self._wait_milliseconds += 1

    def decrease_wait_milliseconds(self) -> None:
        if self._wait_milliseconds >= 2:
            self._wait_milliseconds -= 1

    def aggregate_debugging_information(self) -> Tuple[float, float]:
        return (
            sum(self._process_durations) / len(self._process_durations),
            sum(self._layers_durations) / len(self._layers_durations),
        )

    def update_debugging_information(self) -> None:
        self._process_durations.append(self.process_duration)
        self._layers_durations.append(self.process_layers_duration)

    def clear_debugging_information(self) -> None:
        self._process_durations.clear()
        self._layers_durations.clear()

    def get_logging_information(self) -> str:
        buffer = StringIO()
        buffer.write(f"Original source: {self._source}")
        buffer.write(f"\nRecord destination: {self._output}")
        buffer.write(f"\nPreview mode: {self._preview}")
        buffer.write(f"\nPreview scale: {self._preview_scale}")
        buffer.write(f"\nPreview scale method: {self._preview_scale_method}")
        buffer.write(f"\nVideo resolution: {self._width}x{self._height}")
        buffer.write(f"\nVideo FPS: {self._fps:.2f}")
        buffer.write(f"\nVideo frame count: {self._frames}")
        buffer.write(f"\nVideo start frame position: {self._start_frame_position}")
        return buffer.getvalue()

    def read_next_frame(self) -> bool:
        assert self._capture is not None
        retval, self._original_frame = self._capture.read()
        if not retval:
            return False

        self._frame_index += 1
        if self._frame_index % self._debugging_step == 0:
            process_avg, layers_avg = self.aggregate_debugging_information()

            message = StringIO()
            message.write(f"Read next frame {self._frame_index}/{self._frames} (")
            message.write(f"process avg: {process_avg:.3f}s, ")
            message.write(f"layers avg: {layers_avg:.3f}s)")

            self._layers_durations.append(self.process_layers_duration)
            self.logger.debug(message.getvalue())

        return True

    def read_prev_frame(self) -> bool:
        self.move_frame_position(self._frame_index - 2)
        return self.read_next_frame()

    def move_frame_position(self, frame_index: int) -> None:
        assert self._capture is not None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._frame_index = frame_index

    def process_next_frame(self) -> None:
        begin = datetime.now()
        try:
            self._processed_frame = self.on_frame(self._original_frame.copy())
        finally:
            self._process_duration = (datetime.now() - begin).total_seconds()

        if isclose(self._preview_scale, 1.0):
            self._resized_preview = self._processed_frame.copy()
        else:
            self._resized_preview = cv2.resize(
                self._processed_frame,
                dsize=(0, 0),
                fx=self._preview_scale,  # noqa
                fy=self._preview_scale,  # noqa
                interpolation=self._preview_scale_method,
            )

        gray_processed_frame = len(self._processed_frame.shape) == 2
        if gray_processed_frame:
            self._record_frame = cv2.cvtColor(self._processed_frame, cv2.COLOR_GRAY2BGR)
        else:
            self._record_frame = self._processed_frame.copy()

    def process_key_events(self) -> bool:
        keycode = cv2.waitKeyEx(self._wait_milliseconds)
        if keycode == KEYCODE_NULL:
            return True

        # Process user events with priority.
        # And if it consumed the event with a return value of `True`,
        # It doesn't handle default events.
        if self.on_keydown(keycode):
            return True

        if keycode in self._shortcut_map:
            self._shortcut_map[keycode](keycode)
            return True

        if self._has_arrow_keys:
            if keycode == self._highgui_keys[HighGuiKeyCode.ARROW_UP]:
                self.on_keydown_param_prev(keycode)
                return True
            elif keycode == self._highgui_keys[HighGuiKeyCode.ARROW_DOWN]:
                self.on_keydown_param_next(keycode)
                return True
            elif keycode == self._highgui_keys[HighGuiKeyCode.ARROW_LEFT]:
                self.on_keydown_param_down(keycode)
                return True
            elif keycode == self._highgui_keys[HighGuiKeyCode.ARROW_RIGHT]:
                self.on_keydown_param_up(keycode)
                return True

        self.logger.debug(f"Unmatched keycode: {keycode}")
        return True

    def on_keydown_quit(self, keycode: int) -> None:
        raise InterruptedError("Quit key detected")

    def on_keydown_play(self, keycode: int) -> None:
        assert keycode > 0
        self._play = not self._play
        state_text = "played" if self._play else "stopped"
        self.logger.info(f"The video has been {state_text}")

    def on_keydown_help(self, keycode: int) -> None:
        assert keycode > 0
        self._help = not self._help
        popup_state = "Show" if self._help else "Hide"
        self.logger.info(f"{popup_state} help popup")

    def on_keydown_snapshot(self, keycode: int) -> None:
        assert keycode > 0

        # if not self._output:
        #     raise AttributeError("The '--output' argument is required")
        # output_dir = os.path.basename(os.path.abspath(self._output))

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_basename = os.path.basename(self._source)
        output_dirname = f"{source_basename}-f{self._frame_index}-{now}"
        prefix = os.path.join(os.getcwd(), output_dirname)
        image_ext = ".png"

        if not os.path.isdir(prefix):
            os.mkdir(prefix)

        self.logger.debug(f"Saving all layer snapshots as '{prefix}' directory ...")

        for layer_index in range(self.length_layers):
            prev_path = f"{prefix}/layer{layer_index}-prev{image_ext}"
            next_path = f"{prefix}/layer{layer_index}-next{image_ext}"
            cv2.imwrite(prev_path, self.get_layer_prev_frame(layer_index))
            cv2.imwrite(next_path, self.get_layer_next_frame(layer_index))

        cv2.imwrite(f"{prefix}/original{image_ext}", self._original_frame)
        cv2.imwrite(f"{prefix}/processed{image_ext}", self._processed_frame)
        cv2.imwrite(f"{prefix}/resized{image_ext}", self._resized_preview)
        cv2.imwrite(f"{prefix}/record{image_ext}", self._record_frame)
        cv2.imwrite(f"{prefix}/preview{image_ext}", self.preview_frame)

        self.logger.info(
            f"Successful saving all layer snapshots as '{prefix}' directory!"
        )

    def on_keydown_wait_up(self, keycode: int) -> None:
        assert keycode > 0
        self.increase_wait_milliseconds()
        self.logger.info(f"Increase wait milliseconds to {self._wait_milliseconds}ms")

    def on_keydown_wait_down(self, keycode: int) -> None:
        assert keycode > 0
        self.decrease_wait_milliseconds()
        self.logger.info(f"Decrease wait milliseconds to {self._wait_milliseconds}ms")

    def on_keydown_layer_select(self, keycode: int) -> None:
        assert keycode > 0
        self.change_layer(keycode - ord("0"))
        self.logging_current_layer_information()

    def on_keydown_layer_next(self, keycode: int) -> None:
        assert keycode > 0
        self.change_next_layer()
        self.logging_current_layer_information()

    def on_keydown_layer_prev(self, keycode: int) -> None:
        assert keycode > 0
        self.change_prev_layer()
        self.logging_current_layer_information()

    def on_keydown_layer_last(self, keycode: int) -> None:
        assert keycode > 0
        self.change_last_layer()
        self.logger.info("Change last layer")

    def on_keydown_frame_next(self, keycode: int) -> None:
        assert keycode > 0
        if not self.read_next_frame():
            raise InterruptedError("Failed to read the next frame")

    def on_keydown_frame_prev(self, keycode: int) -> None:
        assert keycode > 0
        if not self.read_prev_frame():
            raise InterruptedError("Failed to read the prev frame")

    def on_keydown_param_prev(self, keycode: int) -> None:
        assert keycode > 0
        if self.is_last_layer:
            raise IndexError("The current layer is the last layer")

        self.current_layer.prev_param_cursor()
        self.logger.info("Change prev param cursor")
        self.logging_current_param_cursor()

    def on_keydown_param_next(self, keycode: int) -> None:
        assert keycode > 0
        if self.is_last_layer:
            raise IndexError("The current layer is the last layer")

        self.current_layer.next_param_cursor()
        self.logger.info("Change next param cursor")
        self.logging_current_param_cursor()

    def on_keydown_param_up(self, keycode: int) -> None:
        assert keycode > 0
        if self.is_last_layer:
            raise IndexError("The current layer is the last layer")

        self.current_layer.increase_at_param_cursor()
        self.logger.info("Increase value at param cursor")
        self.logging_current_param_cursor()

    def on_keydown_param_down(self, keycode: int) -> None:
        assert keycode > 0
        if self.is_last_layer:
            raise IndexError("The current layer is the last layer")

        self.current_layer.decrease_at_param_cursor()
        self.logger.info("Decrease value at param cursor")
        self.logging_current_param_cursor()

    @overrides
    def on_create(self) -> None:
        pass

    @overrides
    def on_destroy(self) -> None:
        pass

    @overrides
    def on_frame(self, frame: NDArray) -> NDArray:
        return self.call_on_frame_with_layers(frame)

    @overrides
    def on_keydown(self, keycode: int) -> bool:
        return self.call_on_keydown_with_current_layer(keycode)

    @overrides
    def on_mouse(self, event: MouseEvent, x: int, y: int, flags: EventFlags) -> None:
        self.call_on_mouse_with_current_layer(event, x, y, flags)

    def _init_capture(self) -> None:
        if not os.path.isfile(self._source):
            raise FileNotFoundError(f"The `{self._source}` can only be of file type")
        if not os.access(self._source, os.R_OK):
            raise PermissionError(f"You need read access to the `{self._source}` file")

        self._capture = cv2.VideoCapture(self._source)

        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open file `{self._source}`")

        self._width = floor(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = floor(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._frames = floor(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if self._width < 1:
            raise RuntimeError("Invalid source video's width")
        if self._height < 1:
            raise RuntimeError("Invalid source video's height")
        if self._fps < 1:
            raise RuntimeError("Invalid source video's FPS")
        if self._frames < 1:
            raise RuntimeError("Invalid source video's frame count")

    def _init_writer(self) -> None:
        if not self._output:
            return

        output = self._output
        fps = self._fps
        width = self._width
        height = self._height
        fourcc = self._fourcc

        self._writer = cv2.VideoWriter(output, fourcc, fps, (width, height))
        assert self._writer is not None

        if not self._writer.isOpened():
            raise RuntimeError("A Video Writer was created but not opened")

    def _run_prefix(self) -> None:
        self.on_create()

        for layer in self._layers:
            layer.on_create()

        self._init_capture()
        self._init_writer()
        self.logger.info(self.get_logging_information())
        self.call_init_params_with_layers()

        assert self._capture is not None
        assert self._capture.isOpened()

        if self._start_frame_position:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame_position)

        def _mouse_cb(event: int, x: int, y: int, flags: int, param: Any) -> None:
            assert param is None
            self.on_mouse(MouseEvent(event), x, y, flags)

        if self._preview:
            title = self._title
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(title, _mouse_cb, None)  # type: ignore[attr-defined]

    def _run_release(self) -> None:
        for layer in self._layers:
            layer.on_destroy()

        self.on_destroy()

        assert self._capture is not None
        self._capture.release()

        if self._writer is not None:
            self._writer.release()
        if self._preview:
            cv2.destroyWindow(self._title)
        self.logger.debug("Done")

    def _run_loop(self) -> bool:
        assert self._capture is not None
        if not self._capture.isOpened():
            self.logger.error("Source video is not opened")
            return False

        if self._play:
            if not self.read_next_frame():
                self.logger.error(f"Failed to read the {self._frame_index}th frame")
                return False

        self.process_next_frame()
        self.update_debugging_information()

        if self._writer is not None:
            assert self._writer.isOpened()
            self._writer.write(self._record_frame)

        if self._preview:
            cv2.imshow(self._title, self.preview_frame)

        try:
            if not self.process_key_events():
                self.logger.warning("Exit key detected")
                return False
        except InterruptedError:
            self.logger.warning("Interrupt detected")
            return False
        except BaseException as e:
            self.logger.error(e)

        if self._shutdown:
            self.logger.warning("Enabled shutdown flag")
            return False

        return True

    def run(self) -> None:
        self._run_prefix()
        try:
            if not self._play:
                if not self.read_next_frame():
                    self.logger.error("Failed to read the first frame")
                    return

            while self._run_loop():
                pass
        finally:
            self._run_release()

# cvlayer

[![PyPI](https://img.shields.io/pypi/v/cvlayer?style=flat-square)](https://pypi.org/project/cvlayer/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cvlayer?style=flat-square)
[![GitHub](https://img.shields.io/github/license/osom8979/cvlayer?style=flat-square)](https://github.com/osom8979/cvlayer/)

OpenCV Layer Helper

## Overview

## Install

Install `cvlayer`:
```shell
pip install cvlayer
```

Install `cvlayer` with `opencv-python`:
```shell
pip install cvlayer[opencv]
```

Install `cvlayer` with `opencv-python-headless`:
```shell
pip install cvlayer[headless]
```

## Usage

### CvLayer

Just inherit `cvlayer.CvLayer`.

```python
from cvlayer import CvLayer


class YourApp(CvLayer):
    def func(self, image):
        self.cvl_cvt_color_bgr2hsv(image)
```

### CvWindow

Just inherit `cvlayer.CvWindow`.

The following sample is a Perspective Transform example:

```python
# -*- coding: utf-8 -*-

from sys import argv, stderr
from sys import exit as sys_exit
from typing import List, Optional

from cvlayer import CvLayer, CvMixin, CvWindow
from cvlayer.typing import PointI
from cvlayer.palette.basic import RED
from numpy.typing import NDArray


class CvTest(CvWindow, CvMixin, CvLayer):
    _points: List[PointI]

    def __init__(self, source: str, destination: Optional[str] = None):
        super().__init__(source, destination)
        left_top = 930, 2750
        left_bottom = 846, 3098
        right_top = 1091, 2750
        right_bottom = 1361, 3098
        self._points = [left_top, left_bottom, right_top, right_bottom]
        self._scale = 2, 4

    def on_frame(self, image: NDArray) -> NDArray:
        with self.layer("select-roi") as layer:
            self.roi = layer.param("roi").build_select_roi().value
            layer.frame = layer.prev_frame

        with self.layer("perspective-points") as layer:
            points = layer.param("pp").build_select_points(self._points).value
            canvas = layer.prev_frame.copy()
            for p in points:
                self.cvl_draw_point(canvas, p, color=RED)
            layer.frame = canvas

        with self.layer("perspective-transform") as layer:
            sw = layer.param("scale-width").build_uint(self._scale[0]).value
            sh = layer.param("scale-height").build_uint(self._scale[1]).value

            xs = list(map(lambda point: point[0], self._points))
            ys = list(map(lambda point: point[1], self._points))
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            width, height = abs(x2 - x1) * sw, abs(y2 - y1) * sh
            roi = 0, 0, width, height
            m = self.cvl_get_perspective_transform_with_quadrilateral(
                left_top=points[0],
                left_bottom=points[1],
                right_top=points[2],
                right_bottom=points[3],
                destination_roi=roi,
            )
            layer.frame = self.cvl_warp_perspective(image, m, (width, height))

        with self.layer("hsv") as layer:
            layer.frame = hsv = self.cvl_cvt_color_bgr2hsv(layer.prev_frame)
        with self.layer("hsv-h") as layer:
            layer.frame = h = hsv[:, :, 0]
        with self.layer("hsv-s") as layer:
            layer.frame = s = hsv[:, :, 1]
        with self.layer("hsv-v") as layer:
            layer.frame = v = hsv[:, :, 2]

        assert h is not None
        assert s is not None
        assert v is not None

        self.cvm_gaussian_blur("v-blur", (3, 19), 0.0, 7.0)
        self.cvm_threshold_binary("v-thresh", 230)

        return self.last_frame


def test_main(*args) -> None:
    source = args[1]
    destination = args[2] if len(args) >= 3 else None

    try:
        app = CvTest(source, destination)
        app.run()
    except Exception as e:
        print(e, file=stderr)
        sys_exit(1)
    else:
        sys_exit(0)


if __name__ == "__main__":
    test_main(*argv)
```

## License

See the [LICENSE](./LICENSE) file for details. In summary,
**cvlayer** is licensed under the **MIT license**.

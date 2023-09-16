# cvlayer

[![PyPI](https://img.shields.io/pypi/v/cvlayer?style=flat-square)](https://pypi.org/project/cvlayer/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cvlayer?style=flat-square)
[![GitHub](https://img.shields.io/github/license/osom8979/cvlayer?style=flat-square)](https://github.com/osom8979/cvlayer/)

OpenCV Layer Helper

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

```python
from sys import argv

from cvlayer import CvLayer, CvWindow
from cvlayer.palette.basic import RED


class YourWindow(CvWindow, CvLayer):
    def __init__(self, source: str):
        super().__init__(source)

    def on_frame(self, image):
        with self.layer("hsv") as layer:
            layer.frame = hsv = self.cvl_cvt_color_bgr2hsv(image)

        with self.layer("select-hsv-channel") as layer:
            i = layer.param("i").build_unsigned(0, max_value=2)
            layer.frame = channel = hsv[:, :, int(i)].copy()

        with self.layer("select-roi") as layer:
            roi = layer.param("roi").build_select_roi()
            self.plot_roi = roi.value
            layer.frame = hsv.copy()
            self.cvl_draw_rectangle(layer.frame, roi.value, color=RED)

        with self.layer("select-points") as layer:
            points = layer.param("points").build_select_points()
            layer.frame = select_points = hsv.copy()
            for x, y in points.value:
                self.cvl_draw_crosshair_point(select_points, x, y, color=RED)

        return channel


if __name__ == "__main__":
    YourWindow(argv[1]).run()
```

## License

See the [LICENSE](./LICENSE) file for details. In summary,
**cvlayer** is licensed under the **MIT license**.

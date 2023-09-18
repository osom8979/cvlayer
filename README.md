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
            layer.frame = self.cvl_cvt_color_bgr2hsv(layer.prev_frame)

        with self.layer("select-hsv-channel") as layer:
            i = layer.param("i").build_unsigned(0, max_value=2).value
            layer.frame = channel = layer.prev_frame[:, :, i].copy()

        with self.layer("select-roi") as layer:
            self.plot_roi = roi = layer.param("roi").build_select_roi().value
            layer.frame = image.copy()
            self.cvl_draw_rectangle(layer.frame, roi, color=RED)

        with self.layer("select-points") as layer:
            points = layer.param("points").build_select_points().value
            layer.frame = select_points = image.copy()
            for x, y in points:
                self.cvl_draw_crosshair_point(select_points, x, y, color=RED)

        return channel


if __name__ == "__main__":
    YourWindow(argv[1]).run()
```

### Samples

```python
from cvlayer import CvLayer, CvWindow

class YourWindow(CvWindow, CvLayer):
    def on_frame(self, image):
        with self.layer("morphology_ex") as layer:
            ksize = layer.param("ksize").build_unsigned(3, 3).value
            i = layer.param("iter").build_unsigned(1, 1).value
            m = self.cvl_get_structuring_element(MorphMethod.ELLIPSE, (ksize, ksize))
            src = layer.prev_frame
            opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, m, iterations=i)
            layer.frame = opening
```

```python
from cvlayer import CvLayer, CvWindow

class YourWindow(CvWindow, CvLayer):
    def on_frame(self, image):
        with self.layer("dilate") as layer:
            ksize = layer.param("ksize").build_unsigned(3, 3).value
            i = layer.param("iter").build_unsigned(1, 1).value
            m = self.cvl_get_structuring_element(MorphMethod.ELLIPSE, (ksize, ksize))
            layer.frame = dilate = self.cvl_dilate(layer.prev_frame, m, iterations=i)
```

```python
from cvlayer import CvLayer, CvWindow

class YourWindow(CvWindow, CvLayer):
    def on_frame(self, image):
        with self.layer("erode") as layer:
            ksize = layer.param("ksize").build_unsigned(3, 3).value
            i = layer.param("iter").build_unsigned(1, 1).value
            m = self.cvl_get_structuring_element(MorphMethod.ELLIPSE, (ksize, ksize))
            layer.frame = erode = self.cvl_erode(layer.prev_frame, m, iterations=i)
```

## License

See the [LICENSE](./LICENSE) file for details. In summary,
**cvlayer** is licensed under the **MIT license**.

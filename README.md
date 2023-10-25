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

from cvlayer import CvLayer, CvMixin, CvWindow


class YourWindow(CvWindow, CvMixin, CvLayer):
    def __init__(self, source: str):
        super().__init__(source)

    def on_frame(self, image):
        with self.layer("select-roi") as layer:
            self.roi = layer.param("roi").build_select_roi().value
            layer.frame = layer.prev_frame

        with self.layer("hsv") as layer:
            layer.frame = self.cvl_cvt_color_bgr2hsv(layer.prev_frame)

        return self.last_frame


def main(source: str):
    YourWindow(source).run()


if __name__ == "__main__":
    main(argv[1])
```

## License

See the [LICENSE](./LICENSE) file for details. In summary,
**cvlayer** is licensed under the **MIT license**.

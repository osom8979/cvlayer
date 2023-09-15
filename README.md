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


class YourWindow(CvWindow, CvLayer):
    def __init__(self, source: str):
        super().__init__(source)

    def on_frame(self, image):
        with self.layer("hsv") as layer:
            hsv = layer.frame = self.cvl_cvt_color_bgr2hsv(image)

        with self.layer("select-hsv-channel") as layer:
            i = layer.param("i").build_unsigned(0, max_value=2)
            channel = layer.frame = hsv[:, :, int(i)].copy()

        return channel


if __name__ == "__main__":
    YourWindow(argv[1]).run()
```

## License

See the [LICENSE](./LICENSE) file for details. In summary,
**cvlayer** is licensed under the **MIT license**.

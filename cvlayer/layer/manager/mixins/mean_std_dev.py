# -*- coding: utf-8 -*-

from io import StringIO
from math import floor
from typing import List, NamedTuple, Optional

from numpy import uint8, zeros
from numpy.typing import NDArray

from cvlayer.cv.basic import mean_std_dev
from cvlayer.cv.bitwise import bitwise_and, bitwise_not
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class MeanStdDevResult(NamedTuple):
    light_mean: List[float]
    light_stddev: List[float]

    dark_mean: List[float]
    dark_stddev: List[float]

    @property
    def avg_light_mean(self) -> float:
        return sum(self.light_mean) / len(self.light_mean)

    @property
    def avg_light_stddev(self) -> float:
        return sum(self.light_stddev) / len(self.light_stddev)

    @property
    def avg_dark_mean(self) -> float:
        return sum(self.dark_mean) / len(self.dark_mean)

    @property
    def avg_dark_stddev(self) -> float:
        return sum(self.dark_stddev) / len(self.dark_stddev)


def _stat(values: List[float]) -> str:
    if len(values) == 0:
        return "[]"

    buffer = StringIO()
    buffer.write(f"[{floor(values[0])}")
    for val in values[1:]:
        buffer.write(f",{floor(val)}")
    buffer.write("]")
    return buffer.getvalue()


class CvmMeanStdDev(LayerManagerMixinBase):
    def cvm_mean_std_dev(
        self,
        name: str,
        mask: Optional[NDArray] = None,
        draw_light=True,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame
            light = mask if mask is not None else zeros((src.shape[0:2]), dtype=uint8)
            dark = bitwise_not(light)

            light_stat = mean_std_dev(src, light)
            dark_stat = mean_std_dev(src, dark)

            light_mean = light_stat.mean.flatten().tolist()
            light_stddev = light_stat.stddev.flatten().tolist()
            dark_mean = dark_stat.mean.flatten().tolist()
            dark_stddev = dark_stat.stddev.flatten().tolist()

            dl = layer.param("draw-light").build_bool(draw_light).value
            layer.param("light-mean").build_readonly([], _stat).value = light_mean
            layer.param("light-stddev").build_readonly([], _stat).value = light_stddev
            layer.param("dark-mean").build_readonly([], _stat).value = dark_mean
            layer.param("dark-stddev").build_readonly([], _stat).value = dark_stddev

            result = MeanStdDevResult(light_mean, light_stddev, dark_mean, dark_stddev)

            layer.frame = bitwise_and(src, src, light if dl else dark)
            layer.data = result

        return src, result

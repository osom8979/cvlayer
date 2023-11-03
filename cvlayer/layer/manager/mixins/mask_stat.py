# -*- coding: utf-8 -*-

from io import StringIO
from typing import List, NamedTuple, Optional

from numpy import uint8, zeros
from numpy.typing import NDArray

from cvlayer.cv.basic import mean_std_dev
from cvlayer.cv.bitwise import bitwise_not
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


class MaskStatResult(NamedTuple):
    light_mean: List[float]
    light_stddev: List[float]
    dark_mean: List[float]
    dark_stddev: List[float]


def _stat(values: List[float]) -> str:
    if len(values) == 0:
        return "[]/sum=0"

    buffer = StringIO()
    buffer.write(f"[{values[0]:.2f}")
    for val in values[1:]:
        buffer.write(f",{val:.2f}")
    buffer.write(f"]/sum={sum(values):.2f}")
    return buffer.getvalue()


class CvmMaskStat(LayerManagerMixinBase):
    def cvm_mask_stat(
        self,
        name: str,
        mask: Optional[NDArray] = None,
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

            layer.param("light-mean").build_readonly([], _stat).value = light_mean
            layer.param("light-stddev").build_readonly([], _stat).value = light_stddev
            layer.param("dark-mean").build_readonly([], _stat).value = dark_mean
            layer.param("dark-stddev").build_readonly([], _stat).value = dark_stddev

            result = MaskStatResult(light_mean, light_stddev, dark_mean, dark_stddev)
            layer.frame = src
            layer.data = result

        return src, result

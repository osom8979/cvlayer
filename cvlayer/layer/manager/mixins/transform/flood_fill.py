# -*- coding: utf-8 -*-

from typing import Optional, Sequence

from numpy.typing import NDArray

from cvlayer.cv.transform.flood_fill import FloodFillFlag, flood_fill
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase
from cvlayer.typing import PointN


class CvmTransformFloodFill(LayerManagerMixinBase):
    def cvm_flood_fill(
        self,
        name: str,
        seed: PointN,
        mask: Optional[NDArray] = None,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            src = frame if frame is not None else layer.prev_frame

            fill_color: Sequence[int]
            lower_diff: Sequence[int]
            upper_diff: Sequence[int]

            if len(src.shape) == 2:
                fv = layer.param("fill-v").build_uint(255, 0, 255).value
                lv = layer.param("lower-v").build_uint(4, 0, 255).value
                uv = layer.param("upper-v").build_uint(4, 0, 255).value

                fill_color = (fv,)
                lower_diff = (lv,)
                upper_diff = (uv,)
            elif len(src.shape) == 3 and src.shape[2] == 3:
                fb = layer.param("fill-b").build_uint(0, 0, 255).value
                fg = layer.param("fill-g").build_uint(255, 0, 255).value
                fr = layer.param("fill-r").build_uint(0, 0, 255).value

                lb = layer.param("lower-b").build_uint(4, 0, 255).value
                lg = layer.param("lower-g").build_uint(4, 0, 255).value
                lr = layer.param("lower-r").build_uint(4, 0, 255).value

                ub = layer.param("upper-b").build_uint(4, 0, 255).value
                ug = layer.param("upper-g").build_uint(4, 0, 255).value
                ur = layer.param("upper-r").build_uint(4, 0, 255).value

                fill_color = fb, fg, fr
                lower_diff = lb, lg, lr
                upper_diff = ub, ug, ur
            else:
                raise ValueError(
                    f"Unsupported src image's shape/dtype: {src.shape}/{src.dtype}"
                )

            seed_point = int(seed[0]), int(seed[1])
            connectivity = layer.param("connectivity").build_list([4, 8]).value
            mask_value = layer.param("mask-value").build_uint(255, 1, 255).value
            fixed_range = layer.param("fixed-range").build_bool(False).value
            mask_only = layer.param("mask-only").build_bool(False).value
            flags = FloodFillFlag(connectivity, mask_value, fixed_range, mask_only)

            result = flood_fill(
                src,
                mask,
                seed_point,
                fill_color,
                lower_diff,
                upper_diff,
                flags,
            )

            layer.param("seed").build_readonly(tuple()).value = seed_point
            layer.param("num").build_readonly(0).value = result.number_of_filled_pixels
            layer.param("roi").build_readonly(tuple()).value = result.roi

            if mask_only:
                layer.frame = result.mask
            else:
                layer.frame = result.image

            layer.data = result.roi

        return result

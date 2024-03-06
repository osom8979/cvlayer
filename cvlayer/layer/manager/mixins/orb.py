# -*- coding: utf-8 -*-

from typing import Optional

from numpy.typing import NDArray

from cvlayer.cv.drawable.keypoint import draw_keypoints
from cvlayer.cv.orb import (
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_FAST_THRESHOLD,
    DEFAULT_FIRST_LEVEL,
    DEFAULT_N_FEATURES,
    DEFAULT_N_LEVELS,
    DEFAULT_PATCH_SIZE,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_SCORE_TYPE,
    DEFAULT_WTA_K,
    Orb,
    OrbScoreType,
)
from cvlayer.cv.types.draw_matches import DrawMatches, normalize_draw_matches
from cvlayer.layer.manager.mixins._base import LayerManagerMixinBase


def _orb_cacher(_old, _new):
    assert isinstance(_new, tuple)
    assert len(_new) == 9

    n_features = _new[0]
    scale_factor = _new[1]
    n_levels = _new[2]
    edge_threshold = _new[3]
    first_level = _new[4]
    wta_k = _new[5]
    score_type = _new[6]
    patch_size = _new[7]
    fast_threshold = _new[8]

    return Orb(
        n_features,
        scale_factor,
        n_levels,
        edge_threshold,
        first_level,
        wta_k,
        score_type,
        patch_size,
        fast_threshold,
    )


class CvmOrb(LayerManagerMixinBase):
    def cvm_orb(
        self,
        name: str,
        n_features=DEFAULT_N_FEATURES,
        scale_factor=DEFAULT_SCALE_FACTOR,
        n_levels=DEFAULT_N_LEVELS,
        edge_threshold=DEFAULT_EDGE_THRESHOLD,
        first_level=DEFAULT_FIRST_LEVEL,
        wta_k=DEFAULT_WTA_K,
        score_type=DEFAULT_SCORE_TYPE,
        patch_size=DEFAULT_PATCH_SIZE,
        fast_threshold=DEFAULT_FAST_THRESHOLD,
        draw_flags=DrawMatches.DEFAULT,
        frame: Optional[NDArray] = None,
    ):
        with self.layer(name) as layer:
            nf = layer.param("n_features").build_uint(n_features).value
            sf = layer.param("scale_factor").build_float(scale_factor).value
            nl = layer.param("n_levels").build_uint(n_levels).value
            et = layer.param("edge_threshold").build_uint(edge_threshold).value
            fl = layer.param("first_level").build_uint(first_level).value
            wk = layer.param("wta_k").build_uint(wta_k).value
            st = layer.param("score_type").build_enum(score_type).value
            ps = layer.param("patch_size").build_uint(patch_size).value
            ft = layer.param("fast_threshold").build_uint(fast_threshold).value
            df = layer.param("draw_flags").build_enum(draw_flags).value
            orb_param = layer.param("orb").build_readonly(tuple(), cacher=_orb_cacher)
            if orb_param.value != (nf, sf, nl, et, fl, wk, st, ps, ft):
                orb_param.value = (nf, sf, nl, et, fl, wk, st, ps, ft)
            orb = orb_param.cache
            src = frame if frame is not None else layer.prev_frame
            keypoints, descriptors = orb.detect_and_compute(src)
            layer.frame = draw_keypoints(
                src,
                keypoints,
                normalize_draw_matches(df),
            )
            layer.data = keypoints, descriptors
        return keypoints, descriptors

    def cvm_orb_harris_score(
        self,
        name: str,
        n_features=DEFAULT_N_FEATURES,
        scale_factor=DEFAULT_SCALE_FACTOR,
        n_levels=DEFAULT_N_LEVELS,
        edge_threshold=DEFAULT_EDGE_THRESHOLD,
        first_level=DEFAULT_FIRST_LEVEL,
        wta_k=DEFAULT_WTA_K,
        patch_size=DEFAULT_PATCH_SIZE,
        fast_threshold=DEFAULT_FAST_THRESHOLD,
        draw_flags=DrawMatches.DEFAULT,
        frame: Optional[NDArray] = None,
    ):
        return self.cvm_orb(
            name=name,
            n_features=n_features,
            scale_factor=scale_factor,
            n_levels=n_levels,
            edge_threshold=edge_threshold,
            first_level=first_level,
            wta_k=wta_k,
            score_type=OrbScoreType.HARRIS_SCORE,
            patch_size=patch_size,
            fast_threshold=fast_threshold,
            draw_flags=draw_flags,
            frame=frame,
        )

    def cvm_orb_fast_score(
        self,
        name: str,
        n_features=DEFAULT_N_FEATURES,
        scale_factor=DEFAULT_SCALE_FACTOR,
        n_levels=DEFAULT_N_LEVELS,
        edge_threshold=DEFAULT_EDGE_THRESHOLD,
        first_level=DEFAULT_FIRST_LEVEL,
        wta_k=DEFAULT_WTA_K,
        patch_size=DEFAULT_PATCH_SIZE,
        fast_threshold=DEFAULT_FAST_THRESHOLD,
        draw_flags=DrawMatches.DEFAULT,
        frame: Optional[NDArray] = None,
    ):
        return self.cvm_orb(
            name=name,
            n_features=n_features,
            scale_factor=scale_factor,
            n_levels=n_levels,
            edge_threshold=edge_threshold,
            first_level=first_level,
            wta_k=wta_k,
            score_type=OrbScoreType.FAST_SCORE,
            patch_size=patch_size,
            fast_threshold=fast_threshold,
            draw_flags=draw_flags,
            frame=frame,
        )

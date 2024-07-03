# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Tuple

import cv2

from cvlayer.cv.stitching.types import (
    BLEND_KEYS,
    BUNDLE_ADJUSTER_KEYS,
    BUNDLE_ADJUSTER_MAP,
    DEFAULT_ETC_MATCH_CONF,
    DEFAULT_ORB_MATCH_CONF,
    ESTIMATOR_KEYS,
    ESTIMATOR_MAP,
    EXPOS_COMP_KEYS,
    EXPOSURE_COMPENSATOR_MAP,
    FEATURES_FINDER_KEYS,
    FEATURES_FINDER_MAP,
    FEATURES_FINDER_ORB,
    MATCHER_AFFINE,
    MATCHER_KEYS,
    SEAM_FINDER_KEYS,
    SEAM_FINDER_MAP,
    STITCHER_MODE_KEYS,
    STITCHER_MODE_MAP,
    TIMELAPSE_AS_IS,
    TIMELAPSE_CROP,
    TIMELAPSE_KEYS,
    WARP_KEYS,
    WAVE_CORRECT_KEYS,
    WAVE_CORRECT_MAP,
)


@dataclass
class StitcherProps:
    stitcher_mode_index: int = 0

    work_size: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    seam_size: Tuple[int, int] = field(default_factory=lambda: (0, 0))

    # Details parameters
    work_mega_pixel: float = 0.6
    features_finder_index: int = 0
    matcher_index: int = 0
    estimator_index: int = 0
    match_conf: float = 0.3
    conf_thresh: float = 0.1
    bundle_adjuster_index: int = 0
    ba_refine_mask: str = "xxxxx"
    wave_correct_index: int = 0
    warp_index: int = 0
    seam_mega_pixel: float = 0.1
    seam_find_index: int = 0
    compose_mega_pixel: float = -1.0
    expos_comp_index: int = 0
    expos_comp_nr_feeds: int = 1
    expos_comp_nr_filtering: float = 2.0
    expos_comp_block_size: int = 32
    blend_index: int = 0
    blend_strength: int = 5
    range_width: int = -1
    timelapse_index: int = 0
    use_cuda: bool = False

    @property
    def stitcher_mode(self):
        key = STITCHER_MODE_KEYS[self.stitcher_mode_index]
        return STITCHER_MODE_MAP[key]

    @property
    def expos_comp_type(self):
        key = EXPOS_COMP_KEYS[self.expos_comp_index]
        return EXPOSURE_COMPENSATOR_MAP[key]

    @property
    def bundle_adjuster(self):
        key = BUNDLE_ADJUSTER_KEYS[self.bundle_adjuster_index]
        return BUNDLE_ADJUSTER_MAP[key]

    @property
    def features_finder(self):
        key = FEATURES_FINDER_KEYS[self.features_finder_index]
        return FEATURES_FINDER_MAP[key]

    @property
    def seam_find(self):
        key = SEAM_FINDER_KEYS[self.seam_find_index]
        return SEAM_FINDER_MAP[key]

    @property
    def estimator(self):
        key = ESTIMATOR_KEYS[self.estimator_index]
        return ESTIMATOR_MAP[key]

    @property
    def wave_correct(self):
        key = WAVE_CORRECT_KEYS[self.wave_correct_index]
        return WAVE_CORRECT_MAP[key]

    @property
    def matcher_key(self):
        return MATCHER_KEYS[self.matcher_index]

    @property
    def warp_key(self):
        return WARP_KEYS[self.warp_index]

    @property
    def blend_key(self):
        return BLEND_KEYS[self.blend_index]

    @property
    def timelapse(self):
        return TIMELAPSE_KEYS[self.timelapse_index]

    @property
    def adjusted_match_conf(self):
        if self.match_conf <= 0.0:
            ff_name = FEATURES_FINDER_KEYS[self.features_finder_index].upper()
            if ff_name == FEATURES_FINDER_ORB:
                return DEFAULT_ORB_MATCH_CONF
            else:
                return DEFAULT_ETC_MATCH_CONF
        else:
            return self.match_conf

    def get_matcher(self):
        use_cuda = self.use_cuda
        matcher_name = self.matcher_key
        match_conf = self.adjusted_match_conf
        range_width = self.range_width

        if matcher_name == MATCHER_AFFINE:
            return cv2.detail.AffineBestOf2NearestMatcher(False, use_cuda, match_conf)
        elif range_width == -1:
            return cv2.detail.BestOf2NearestMatcher(use_cuda, match_conf)
        else:
            return cv2.detail.BestOf2NearestRangeMatcher(
                range_width, use_cuda, match_conf
            )

    def get_compensator(self, use_filter=False):
        if self.expos_comp_type == cv2.detail.ExposureCompensator_CHANNELS:
            return cv2.detail.ChannelsCompensator(self.expos_comp_nr_feeds)
        elif self.expos_comp_type == cv2.detail.ExposureCompensator_CHANNELS_BLOCKS:
            compensator = cv2.detail.BlocksChannelsCompensator(
                self.expos_comp_block_size,
                self.expos_comp_block_size,
                self.expos_comp_nr_feeds,
            )
            if use_filter:
                nr_iterations = int(self.expos_comp_nr_filtering)
                compensator.setNrGainsFilteringIterations(nr_iterations)
            return compensator
        else:
            return cv2.detail.ExposureCompensator.createDefault(self.expos_comp_type)

    def get_timelapse(self):
        if self.timelapse == TIMELAPSE_AS_IS:
            return cv2.detail.Timelapser_AS_IS
        elif self.timelapse == TIMELAPSE_CROP:
            return cv2.detail.Timelapser_CROP
        else:
            assert False, "Inaccessible section"

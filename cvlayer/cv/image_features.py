# -*- coding: utf-8 -*-

from typing import Sequence

import cv2


def serialize_image_features(features: Sequence[cv2.detail.ImageFeatures]):
    result = []

    for f in features:
        keypoints = []

        for kp in f.keypoints:
            o = dict(
                angle=kp.angle,
                class_id=kp.class_id,
                octave=kp.octave,
                pt=[x for x in kp.pt],
                response=kp.response,
                size=kp.size,
            )
            keypoints.append(o)

        result.append(
            dict(
                img_idx=f.img_idx,
                img_size=[x for x in f.img_size],
                keypoints=keypoints,
                descriptors=f.descriptors.get(),
            )
        )

    return result

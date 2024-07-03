# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from cvlayer.cv.stitching.parts import StitcherPart
from cvlayer.cv.stitching.props import StitcherProps


class StitchError(Exception):
    pass


class Stitcher:
    """
    Rotation model images stitcher
    ::
    https://docs.opencv.org/4.x/d2/d8d/classcv_1_1Stitcher.html
    """

    parts: OrderedDict[str, StitcherPart]
    result: Optional[NDArray]

    def __init__(self, props: StitcherProps):
        self.props = props
        self.stitcher = cv2.Stitcher.create(self.props.stitcher_mode)
        self.parts = OrderedDict()
        self.result = None

    def clear_images(self) -> None:
        self.parts.clear()

    def add_image(self, filepath: str) -> StitcherPart:
        result = StitcherPart(filepath)
        self.parts[filepath] = result
        return result

    def remove_image(self, name: str) -> StitcherPart:
        return self.parts.pop(name)

    @property
    def keys(self) -> List[str]:
        return list(self.parts.keys())

    @property
    def images(self) -> List[NDArray]:
        return [part.original for part in self.parts.values()]

    def change_mode(self) -> None:
        self.stitcher = cv2.Stitcher.create(self.props.stitcher_mode)

    def stitch(self) -> NDArray:
        status, self.result = self.stitcher.stitch(self.images)
        if status == cv2.Stitcher_OK:
            return self.result
        elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            raise StitchError(status, "Need more images")
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            raise StitchError(status, "Homography estimate fail")
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            raise StitchError(status, "Camera params adjust fail")
        else:
            assert False, "Inaccessible section"

    def stitch_details(self):
        img_names = self.keys
        work_mega_pixel = self.props.work_mega_pixel
        seam_mega_pixel = self.props.seam_mega_pixel
        compose_mega_pixel = self.props.compose_mega_pixel
        conf_thresh = self.props.conf_thresh
        ba_refine_mask = self.props.ba_refine_mask
        wave_correct = self.props.wave_correct
        warp_type = self.props.warp_key
        blend_type = self.props.blend_key
        blend_strength = self.props.blend_strength
        finder = self.props.features_finder()

        seam_work_aspect = 1
        full_img_sizes = []
        features = []
        images = []

        is_work_scale_set = False
        is_seam_scale_set = False
        is_compose_scale_set = False

        work_scale = 1.0
        seam_scale = 1.0

        for full_img in self.images:
            height = full_img.shape[0]
            width = full_img.shape[1]
            size = width, height
            pixels = height * width

            full_img_sizes.append(size)
            if work_mega_pixel < 0:
                img = full_img
                work_scale = 1.0
                is_work_scale_set = True
            else:
                if is_work_scale_set is False:
                    work_scale = min(1.0, np.sqrt(work_mega_pixel * 1e6 / pixels))
                    is_work_scale_set = True

                img = cv2.resize(
                    src=full_img,
                    dsize=None,
                    fx=work_scale,
                    fy=work_scale,
                    interpolation=cv2.INTER_LINEAR_EXACT,
                )

            if is_seam_scale_set is False:
                if seam_mega_pixel > 0:
                    seam_scale = min(1.0, np.sqrt(seam_mega_pixel * 1e6 / pixels))
                else:
                    seam_scale = 1.0
                seam_work_aspect = seam_scale / work_scale
                is_seam_scale_set = True

            img_feat = cv2.detail.computeImageFeatures2(finder, img)
            features.append(img_feat)
            img = cv2.resize(
                src=full_img,
                dsize=None,
                fx=seam_scale,
                fy=seam_scale,
                interpolation=cv2.INTER_LINEAR_EXACT,
            )
            images.append(img)

        matcher = self.props.get_matcher()
        p = matcher.apply2(features)
        matcher.collectGarbage()

        indices = cv2.detail.leaveBiggestComponent(features, p, conf_thresh)
        img_subset = []
        img_names_subset = []
        full_img_sizes_subset = []
        for i in range(len(indices)):
            img_names_subset.append(img_names[indices[i]])
            img_subset.append(images[indices[i]])
            full_img_sizes_subset.append(full_img_sizes[indices[i]])
        images = img_subset
        img_names = img_names_subset
        full_img_sizes = full_img_sizes_subset
        num_images = len(img_names)
        if num_images < 2:
            raise StitchError(cv2.Stitcher_ERR_NEED_MORE_IMGS, "Need more images")

        estimator = self.props.estimator()
        b, cameras = estimator.apply(features, p, None)
        if not b:
            raise StitchError(
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL,
                "Homography estimation failed",
            )

        for cam in cameras:
            cam.R = cam.R.astype(np.float32)

        adjuster = self.props.bundle_adjuster()
        adjuster.setConfThresh(conf_thresh)
        refine_mask = np.zeros((3, 3), np.uint8)
        if ba_refine_mask[0] == "x":
            refine_mask[0, 0] = 1
        if ba_refine_mask[1] == "x":
            refine_mask[0, 1] = 1
        if ba_refine_mask[2] == "x":
            refine_mask[0, 2] = 1
        if ba_refine_mask[3] == "x":
            refine_mask[1, 1] = 1
        if ba_refine_mask[4] == "x":
            refine_mask[1, 2] = 1
        adjuster.setRefinementMask(refine_mask)
        b, cameras = adjuster.apply(features, p, cameras)
        if not b:
            raise StitchError(
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL,
                "Camera parameters adjusting failed",
            )

        focals = []
        for cam in cameras:
            focals.append(cam.focal)
        focals.sort()

        if len(focals) % 2 == 1:
            warped_image_scale = focals[len(focals) // 2]
        else:
            focal_mid = focals[len(focals) // 2]
            focal_mid_m1 = focals[len(focals) // 2 - 1]
            warped_image_scale = (focal_mid + focal_mid_m1) / 2

        if wave_correct is not None:
            r_mats = []
            for cam in cameras:
                r_mats.append(np.copy(cam.R))
            r_mats = cv2.detail.waveCorrect(r_mats, wave_correct)
            for idx, cam in enumerate(cameras):
                cam.R = r_mats[idx]

        corners = []
        masks_warped = []
        images_warped = []
        sizes = []
        masks = []

        for i in range(0, num_images):
            h = images[i].shape[0]
            w = images[i].shape[1]
            # noinspection PyTypeChecker
            um = cv2.UMat(255 * np.ones((h, w), np.uint8))
            masks.append(um)

        warper = cv2.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)

        for idx in range(0, num_images):
            K = cameras[idx].K().astype(np.float32)
            swa = seam_work_aspect
            K[0, 0] *= swa
            K[0, 2] *= swa
            K[1, 1] *= swa
            K[1, 2] *= swa
            corner, image_wp = warper.warp(
                images[idx], K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT
            )
            corners.append(corner)
            sizes.append((image_wp.shape[1], image_wp.shape[0]))
            images_warped.append(image_wp)
            p, mask_wp = warper.warp(
                masks[idx], K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT
            )
            masks_warped.append(mask_wp.get())

        images_warped_f = []
        for img in images_warped:
            imgf = img.astype(np.float32)
            images_warped_f.append(imgf)

        compensator = self.props.get_compensator()
        compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

        seam_finder = self.props.seam_find
        masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
        compose_scale = 1
        corners = []
        sizes = []
        blender = None

        for idx, name in enumerate(img_names):
            full_img = cv2.imread(name)

            if not is_compose_scale_set:
                if compose_mega_pixel > 0:
                    compose_scale = min(
                        1.0,
                        np.sqrt(
                            compose_mega_pixel
                            * 1e6
                            / (full_img.shape[0] * full_img.shape[1])
                        ),
                    )
                is_compose_scale_set = True
                compose_work_aspect = compose_scale / work_scale
                warped_image_scale *= compose_work_aspect
                warper = cv2.PyRotationWarper(warp_type, warped_image_scale)
                for i in range(0, len(img_names)):
                    cameras[i].focal *= compose_work_aspect
                    cameras[i].ppx *= compose_work_aspect
                    cameras[i].ppy *= compose_work_aspect
                    sz = (
                        int(round(full_img_sizes[i][0] * compose_scale)),
                        int(round(full_img_sizes[i][1] * compose_scale)),
                    )
                    K = cameras[i].K().astype(np.float32)
                    roi = warper.warpRoi(sz, K, cameras[i].R)
                    corners.append(roi[0:2])
                    sizes.append(roi[2:4])
            if abs(compose_scale - 1) > 1e-1:
                img = cv2.resize(
                    src=full_img,
                    dsize=None,
                    fx=compose_scale,
                    fy=compose_scale,
                    interpolation=cv2.INTER_LINEAR_EXACT,
                )
            else:
                img = full_img
            K = cameras[idx].K().astype(np.float32)
            corner, image_warped = warper.warp(
                img, K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT
            )
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(
                mask, K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT
            )
            compensator.apply(idx, corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            dilated_mask = cv2.dilate(masks_warped[idx], None)
            seam_mask = cv2.resize(
                dilated_mask,
                (mask_warped.shape[1], mask_warped.shape[0]),
                0,
                0,
                cv2.INTER_LINEAR_EXACT,
            )
            mask_warped = cv2.bitwise_and(seam_mask, mask_warped)

            if blender is None:
                blender = cv2.detail.Blender.createDefault(cv2.detail.Blender_NO)
                dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
                blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
                if blend_width < 1:
                    blender = cv2.detail.Blender.createDefault(cv2.detail.Blender_NO)
                elif blend_type == "multiband":
                    blender = cv2.detail.MultiBandBlender()
                    blender.setNumBands(
                        (np.log(blend_width) / np.log(2.0) - 1.0).astype(np.int32)
                    )
                elif blend_type == "feather":
                    blender = cv2.detail.FeatherBlender()
                    blender.setSharpness(1.0 / blend_width)
                blender.prepare(dst_sz)

            blender.feed(cv2.UMat(image_warped_s), mask_warped, corners[idx])

        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        # cv2.imwrite(result_name, result)
        # zoom_x = 600.0 / result.shape[1]
        dst = cv2.normalize(
            src=result,
            dst=None,
            alpha=255.0,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        # dst = cv2.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
        # cv2.imshow(result_name, dst)
        self.result = dst
        return dst

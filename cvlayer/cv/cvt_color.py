# -*- coding: utf-8 -*-

from enum import Enum
from functools import partial

import cv2
from numpy.typing import NDArray

assert cv2.COLOR_BGR2YCrCb == cv2.COLOR_BGR2YCR_CB
assert cv2.COLOR_RGB2YCrCb == cv2.COLOR_RGB2YCR_CB
assert cv2.COLOR_YCrCb2BGR == cv2.COLOR_YCR_CB2BGR
assert cv2.COLOR_YCrCb2RGB == cv2.COLOR_YCR_CB2RGB


class CvtColorCode(Enum):
    BGR2BGRA = cv2.COLOR_BGR2BGRA
    RGB2RGBA = cv2.COLOR_RGB2RGBA
    BGRA2BGR = cv2.COLOR_BGRA2BGR
    RGBA2RGB = cv2.COLOR_RGBA2RGB
    BGR2RGBA = cv2.COLOR_BGR2RGBA
    RGB2BGRA = cv2.COLOR_RGB2BGRA
    RGBA2BGR = cv2.COLOR_RGBA2BGR
    BGRA2RGB = cv2.COLOR_BGRA2RGB
    BGR2RGB = cv2.COLOR_BGR2RGB
    RGB2BGR = cv2.COLOR_RGB2BGR
    BGRA2RGBA = cv2.COLOR_BGRA2RGBA
    RGBA2BGRA = cv2.COLOR_RGBA2BGRA
    BGR2GRAY = cv2.COLOR_BGR2GRAY
    RGB2GRAY = cv2.COLOR_RGB2GRAY
    GRAY2BGR = cv2.COLOR_GRAY2BGR
    GRAY2RGB = cv2.COLOR_GRAY2RGB
    GRAY2BGRA = cv2.COLOR_GRAY2BGRA
    GRAY2RGBA = cv2.COLOR_GRAY2RGBA
    BGRA2GRAY = cv2.COLOR_BGRA2GRAY
    RGBA2GRAY = cv2.COLOR_RGBA2GRAY
    BGR2BGR565 = cv2.COLOR_BGR2BGR565
    RGB2BGR565 = cv2.COLOR_RGB2BGR565
    BGR5652BGR = cv2.COLOR_BGR5652BGR
    BGR5652RGB = cv2.COLOR_BGR5652RGB
    BGRA2BGR565 = cv2.COLOR_BGRA2BGR565
    RGBA2BGR565 = cv2.COLOR_RGBA2BGR565
    BGR5652BGRA = cv2.COLOR_BGR5652BGRA
    BGR5652RGBA = cv2.COLOR_BGR5652RGBA
    GRAY2BGR565 = cv2.COLOR_GRAY2BGR565
    BGR5652GRAY = cv2.COLOR_BGR5652GRAY
    BGR2BGR555 = cv2.COLOR_BGR2BGR555
    RGB2BGR555 = cv2.COLOR_RGB2BGR555
    BGR5552BGR = cv2.COLOR_BGR5552BGR
    BGR5552RGB = cv2.COLOR_BGR5552RGB
    BGRA2BGR555 = cv2.COLOR_BGRA2BGR555
    RGBA2BGR555 = cv2.COLOR_RGBA2BGR555
    BGR5552BGRA = cv2.COLOR_BGR5552BGRA
    BGR5552RGBA = cv2.COLOR_BGR5552RGBA
    GRAY2BGR555 = cv2.COLOR_GRAY2BGR555
    BGR5552GRAY = cv2.COLOR_BGR5552GRAY
    BGR2XYZ = cv2.COLOR_BGR2XYZ
    RGB2XYZ = cv2.COLOR_RGB2XYZ
    XYZ2BGR = cv2.COLOR_XYZ2BGR
    XYZ2RGB = cv2.COLOR_XYZ2RGB
    BGR2YCrCb = cv2.COLOR_BGR2YCrCb
    BGR2YCR_CB = cv2.COLOR_BGR2YCR_CB
    RGB2YCrCb = cv2.COLOR_RGB2YCrCb
    RGB2YCR_CB = cv2.COLOR_RGB2YCR_CB
    YCrCb2BGR = cv2.COLOR_YCrCb2BGR
    YCR_CB2BGR = cv2.COLOR_YCR_CB2BGR
    YCrCb2RGB = cv2.COLOR_YCrCb2RGB
    YCR_CB2RGB = cv2.COLOR_YCR_CB2RGB
    BGR2HSV = cv2.COLOR_BGR2HSV
    RGB2HSV = cv2.COLOR_RGB2HSV
    BGR2Lab = cv2.COLOR_BGR2Lab
    BGR2LAB = cv2.COLOR_BGR2LAB
    RGB2Lab = cv2.COLOR_RGB2Lab
    RGB2LAB = cv2.COLOR_RGB2LAB
    BGR2Luv = cv2.COLOR_BGR2Luv
    BGR2LUV = cv2.COLOR_BGR2LUV
    RGB2Luv = cv2.COLOR_RGB2Luv
    RGB2LUV = cv2.COLOR_RGB2LUV
    BGR2HLS = cv2.COLOR_BGR2HLS
    RGB2HLS = cv2.COLOR_RGB2HLS
    HSV2BGR = cv2.COLOR_HSV2BGR
    HSV2RGB = cv2.COLOR_HSV2RGB
    Lab2BGR = cv2.COLOR_Lab2BGR
    LAB2BGR = cv2.COLOR_LAB2BGR
    Lab2RGB = cv2.COLOR_Lab2RGB
    LAB2RGB = cv2.COLOR_LAB2RGB
    Luv2BGR = cv2.COLOR_Luv2BGR
    LUV2BGR = cv2.COLOR_LUV2BGR
    Luv2RGB = cv2.COLOR_Luv2RGB
    LUV2RGB = cv2.COLOR_LUV2RGB
    HLS2BGR = cv2.COLOR_HLS2BGR
    HLS2RGB = cv2.COLOR_HLS2RGB
    BGR2HSV_FULL = cv2.COLOR_BGR2HSV_FULL
    RGB2HSV_FULL = cv2.COLOR_RGB2HSV_FULL
    BGR2HLS_FULL = cv2.COLOR_BGR2HLS_FULL
    RGB2HLS_FULL = cv2.COLOR_RGB2HLS_FULL
    HSV2BGR_FULL = cv2.COLOR_HSV2BGR_FULL
    HSV2RGB_FULL = cv2.COLOR_HSV2RGB_FULL
    HLS2BGR_FULL = cv2.COLOR_HLS2BGR_FULL
    HLS2RGB_FULL = cv2.COLOR_HLS2RGB_FULL
    LBGR2Lab = cv2.COLOR_LBGR2Lab
    LBGR2LAB = cv2.COLOR_LBGR2LAB
    LRGB2Lab = cv2.COLOR_LRGB2Lab
    LRGB2LAB = cv2.COLOR_LRGB2LAB
    LBGR2Luv = cv2.COLOR_LBGR2Luv
    LBGR2LUV = cv2.COLOR_LBGR2LUV
    LRGB2Luv = cv2.COLOR_LRGB2Luv
    LRGB2LUV = cv2.COLOR_LRGB2LUV
    Lab2LBGR = cv2.COLOR_Lab2LBGR
    LAB2LBGR = cv2.COLOR_LAB2LBGR
    Lab2LRGB = cv2.COLOR_Lab2LRGB
    LAB2LRGB = cv2.COLOR_LAB2LRGB
    Luv2LBGR = cv2.COLOR_Luv2LBGR
    LUV2LBGR = cv2.COLOR_LUV2LBGR
    Luv2LRGB = cv2.COLOR_Luv2LRGB
    LUV2LRGB = cv2.COLOR_LUV2LRGB
    BGR2YUV = cv2.COLOR_BGR2YUV
    RGB2YUV = cv2.COLOR_RGB2YUV
    YUV2BGR = cv2.COLOR_YUV2BGR
    YUV2RGB = cv2.COLOR_YUV2RGB
    YUV2RGB_NV12 = cv2.COLOR_YUV2RGB_NV12
    YUV2BGR_NV12 = cv2.COLOR_YUV2BGR_NV12
    YUV2RGB_NV21 = cv2.COLOR_YUV2RGB_NV21
    YUV2BGR_NV21 = cv2.COLOR_YUV2BGR_NV21
    YUV420sp2RGB = cv2.COLOR_YUV420sp2RGB
    YUV420SP2RGB = cv2.COLOR_YUV420SP2RGB
    YUV420sp2BGR = cv2.COLOR_YUV420sp2BGR
    YUV420SP2BGR = cv2.COLOR_YUV420SP2BGR
    YUV2RGBA_NV12 = cv2.COLOR_YUV2RGBA_NV12
    YUV2BGRA_NV12 = cv2.COLOR_YUV2BGRA_NV12
    YUV2RGBA_NV21 = cv2.COLOR_YUV2RGBA_NV21
    YUV2BGRA_NV21 = cv2.COLOR_YUV2BGRA_NV21
    YUV420sp2RGBA = cv2.COLOR_YUV420sp2RGBA
    YUV420SP2RGBA = cv2.COLOR_YUV420SP2RGBA
    YUV420sp2BGRA = cv2.COLOR_YUV420sp2BGRA
    YUV420SP2BGRA = cv2.COLOR_YUV420SP2BGRA
    YUV2RGB_YV12 = cv2.COLOR_YUV2RGB_YV12
    YUV2BGR_YV12 = cv2.COLOR_YUV2BGR_YV12
    YUV2RGB_IYUV = cv2.COLOR_YUV2RGB_IYUV
    YUV2BGR_IYUV = cv2.COLOR_YUV2BGR_IYUV
    YUV2RGB_I420 = cv2.COLOR_YUV2RGB_I420
    YUV2BGR_I420 = cv2.COLOR_YUV2BGR_I420
    YUV420p2RGB = cv2.COLOR_YUV420p2RGB
    YUV420P2RGB = cv2.COLOR_YUV420P2RGB
    YUV420p2BGR = cv2.COLOR_YUV420p2BGR
    YUV420P2BGR = cv2.COLOR_YUV420P2BGR
    YUV2RGBA_YV12 = cv2.COLOR_YUV2RGBA_YV12
    YUV2BGRA_YV12 = cv2.COLOR_YUV2BGRA_YV12
    YUV2RGBA_IYUV = cv2.COLOR_YUV2RGBA_IYUV
    YUV2BGRA_IYUV = cv2.COLOR_YUV2BGRA_IYUV
    YUV2RGBA_I420 = cv2.COLOR_YUV2RGBA_I420
    YUV2BGRA_I420 = cv2.COLOR_YUV2BGRA_I420
    YUV420p2RGBA = cv2.COLOR_YUV420p2RGBA
    YUV420P2RGBA = cv2.COLOR_YUV420P2RGBA
    YUV420p2BGRA = cv2.COLOR_YUV420p2BGRA
    YUV420P2BGRA = cv2.COLOR_YUV420P2BGRA
    YUV2GRAY_420 = cv2.COLOR_YUV2GRAY_420
    YUV2GRAY_NV21 = cv2.COLOR_YUV2GRAY_NV21
    YUV2GRAY_NV12 = cv2.COLOR_YUV2GRAY_NV12
    YUV2GRAY_YV12 = cv2.COLOR_YUV2GRAY_YV12
    YUV2GRAY_IYUV = cv2.COLOR_YUV2GRAY_IYUV
    YUV2GRAY_I420 = cv2.COLOR_YUV2GRAY_I420
    YUV420sp2GRAY = cv2.COLOR_YUV420sp2GRAY
    YUV420SP2GRAY = cv2.COLOR_YUV420SP2GRAY
    YUV420p2GRAY = cv2.COLOR_YUV420p2GRAY
    YUV420P2GRAY = cv2.COLOR_YUV420P2GRAY
    YUV2RGB_UYVY = cv2.COLOR_YUV2RGB_UYVY
    YUV2BGR_UYVY = cv2.COLOR_YUV2BGR_UYVY
    YUV2RGB_Y422 = cv2.COLOR_YUV2RGB_Y422
    YUV2BGR_Y422 = cv2.COLOR_YUV2BGR_Y422
    YUV2RGB_UYNV = cv2.COLOR_YUV2RGB_UYNV
    YUV2BGR_UYNV = cv2.COLOR_YUV2BGR_UYNV
    YUV2RGBA_UYVY = cv2.COLOR_YUV2RGBA_UYVY
    YUV2BGRA_UYVY = cv2.COLOR_YUV2BGRA_UYVY
    YUV2RGBA_Y422 = cv2.COLOR_YUV2RGBA_Y422
    YUV2BGRA_Y422 = cv2.COLOR_YUV2BGRA_Y422
    YUV2RGBA_UYNV = cv2.COLOR_YUV2RGBA_UYNV
    YUV2BGRA_UYNV = cv2.COLOR_YUV2BGRA_UYNV
    YUV2RGB_YUY2 = cv2.COLOR_YUV2RGB_YUY2
    YUV2BGR_YUY2 = cv2.COLOR_YUV2BGR_YUY2
    YUV2RGB_YVYU = cv2.COLOR_YUV2RGB_YVYU
    YUV2BGR_YVYU = cv2.COLOR_YUV2BGR_YVYU
    YUV2RGB_YUYV = cv2.COLOR_YUV2RGB_YUYV
    YUV2BGR_YUYV = cv2.COLOR_YUV2BGR_YUYV
    YUV2RGB_YUNV = cv2.COLOR_YUV2RGB_YUNV
    YUV2BGR_YUNV = cv2.COLOR_YUV2BGR_YUNV
    YUV2RGBA_YUY2 = cv2.COLOR_YUV2RGBA_YUY2
    YUV2BGRA_YUY2 = cv2.COLOR_YUV2BGRA_YUY2
    YUV2RGBA_YVYU = cv2.COLOR_YUV2RGBA_YVYU
    YUV2BGRA_YVYU = cv2.COLOR_YUV2BGRA_YVYU
    YUV2RGBA_YUYV = cv2.COLOR_YUV2RGBA_YUYV
    YUV2BGRA_YUYV = cv2.COLOR_YUV2BGRA_YUYV
    YUV2RGBA_YUNV = cv2.COLOR_YUV2RGBA_YUNV
    YUV2BGRA_YUNV = cv2.COLOR_YUV2BGRA_YUNV
    YUV2GRAY_UYVY = cv2.COLOR_YUV2GRAY_UYVY
    YUV2GRAY_YUY2 = cv2.COLOR_YUV2GRAY_YUY2
    YUV2GRAY_Y422 = cv2.COLOR_YUV2GRAY_Y422
    YUV2GRAY_UYNV = cv2.COLOR_YUV2GRAY_UYNV
    YUV2GRAY_YVYU = cv2.COLOR_YUV2GRAY_YVYU
    YUV2GRAY_YUYV = cv2.COLOR_YUV2GRAY_YUYV
    YUV2GRAY_YUNV = cv2.COLOR_YUV2GRAY_YUNV
    RGBA2mRGBA = cv2.COLOR_RGBA2mRGBA
    RGBA2M_RGBA = cv2.COLOR_RGBA2M_RGBA
    mRGBA2RGBA = cv2.COLOR_mRGBA2RGBA
    M_RGBA2RGBA = cv2.COLOR_M_RGBA2RGBA
    RGB2YUV_I420 = cv2.COLOR_RGB2YUV_I420
    BGR2YUV_I420 = cv2.COLOR_BGR2YUV_I420
    RGB2YUV_IYUV = cv2.COLOR_RGB2YUV_IYUV
    BGR2YUV_IYUV = cv2.COLOR_BGR2YUV_IYUV
    RGBA2YUV_I420 = cv2.COLOR_RGBA2YUV_I420
    BGRA2YUV_I420 = cv2.COLOR_BGRA2YUV_I420
    RGBA2YUV_IYUV = cv2.COLOR_RGBA2YUV_IYUV
    BGRA2YUV_IYUV = cv2.COLOR_BGRA2YUV_IYUV
    RGB2YUV_YV12 = cv2.COLOR_RGB2YUV_YV12
    BGR2YUV_YV12 = cv2.COLOR_BGR2YUV_YV12
    RGBA2YUV_YV12 = cv2.COLOR_RGBA2YUV_YV12
    BGRA2YUV_YV12 = cv2.COLOR_BGRA2YUV_YV12
    BayerBG2BGR = cv2.COLOR_BayerBG2BGR
    BAYER_BG2BGR = cv2.COLOR_BAYER_BG2BGR
    BayerGB2BGR = cv2.COLOR_BayerGB2BGR
    BAYER_GB2BGR = cv2.COLOR_BAYER_GB2BGR
    BayerRG2BGR = cv2.COLOR_BayerRG2BGR
    BAYER_RG2BGR = cv2.COLOR_BAYER_RG2BGR
    BayerGR2BGR = cv2.COLOR_BayerGR2BGR
    BAYER_GR2BGR = cv2.COLOR_BAYER_GR2BGR
    BayerRGGB2BGR = cv2.COLOR_BayerRGGB2BGR
    BAYER_RGGB2BGR = cv2.COLOR_BAYER_RGGB2BGR
    BayerGRBG2BGR = cv2.COLOR_BayerGRBG2BGR
    BAYER_GRBG2BGR = cv2.COLOR_BAYER_GRBG2BGR
    BayerBGGR2BGR = cv2.COLOR_BayerBGGR2BGR
    BAYER_BGGR2BGR = cv2.COLOR_BAYER_BGGR2BGR
    BayerGBRG2BGR = cv2.COLOR_BayerGBRG2BGR
    BAYER_GBRG2BGR = cv2.COLOR_BAYER_GBRG2BGR
    BayerRGGB2RGB = cv2.COLOR_BayerRGGB2RGB
    BAYER_RGGB2RGB = cv2.COLOR_BAYER_RGGB2RGB
    BayerGRBG2RGB = cv2.COLOR_BayerGRBG2RGB
    BAYER_GRBG2RGB = cv2.COLOR_BAYER_GRBG2RGB
    BayerBGGR2RGB = cv2.COLOR_BayerBGGR2RGB
    BAYER_BGGR2RGB = cv2.COLOR_BAYER_BGGR2RGB
    BayerGBRG2RGB = cv2.COLOR_BayerGBRG2RGB
    BAYER_GBRG2RGB = cv2.COLOR_BAYER_GBRG2RGB
    BayerBG2RGB = cv2.COLOR_BayerBG2RGB
    BAYER_BG2RGB = cv2.COLOR_BAYER_BG2RGB
    BayerGB2RGB = cv2.COLOR_BayerGB2RGB
    BAYER_GB2RGB = cv2.COLOR_BAYER_GB2RGB
    BayerRG2RGB = cv2.COLOR_BayerRG2RGB
    BAYER_RG2RGB = cv2.COLOR_BAYER_RG2RGB
    BayerGR2RGB = cv2.COLOR_BayerGR2RGB
    BAYER_GR2RGB = cv2.COLOR_BAYER_GR2RGB
    BayerBG2GRAY = cv2.COLOR_BayerBG2GRAY
    BAYER_BG2GRAY = cv2.COLOR_BAYER_BG2GRAY
    BayerGB2GRAY = cv2.COLOR_BayerGB2GRAY
    BAYER_GB2GRAY = cv2.COLOR_BAYER_GB2GRAY
    BayerRG2GRAY = cv2.COLOR_BayerRG2GRAY
    BAYER_RG2GRAY = cv2.COLOR_BAYER_RG2GRAY
    BayerGR2GRAY = cv2.COLOR_BayerGR2GRAY
    BAYER_GR2GRAY = cv2.COLOR_BAYER_GR2GRAY
    BayerRGGB2GRAY = cv2.COLOR_BayerRGGB2GRAY
    BAYER_RGGB2GRAY = cv2.COLOR_BAYER_RGGB2GRAY
    BayerGRBG2GRAY = cv2.COLOR_BayerGRBG2GRAY
    BAYER_GRBG2GRAY = cv2.COLOR_BAYER_GRBG2GRAY
    BayerBGGR2GRAY = cv2.COLOR_BayerBGGR2GRAY
    BAYER_BGGR2GRAY = cv2.COLOR_BAYER_BGGR2GRAY
    BayerGBRG2GRAY = cv2.COLOR_BayerGBRG2GRAY
    BAYER_GBRG2GRAY = cv2.COLOR_BAYER_GBRG2GRAY
    BayerBG2BGR_VNG = cv2.COLOR_BayerBG2BGR_VNG
    BAYER_BG2BGR_VNG = cv2.COLOR_BAYER_BG2BGR_VNG
    BayerGB2BGR_VNG = cv2.COLOR_BayerGB2BGR_VNG
    BAYER_GB2BGR_VNG = cv2.COLOR_BAYER_GB2BGR_VNG
    BayerRG2BGR_VNG = cv2.COLOR_BayerRG2BGR_VNG
    BAYER_RG2BGR_VNG = cv2.COLOR_BAYER_RG2BGR_VNG
    BayerGR2BGR_VNG = cv2.COLOR_BayerGR2BGR_VNG
    BAYER_GR2BGR_VNG = cv2.COLOR_BAYER_GR2BGR_VNG
    BayerRGGB2BGR_VNG = cv2.COLOR_BayerRGGB2BGR_VNG
    BAYER_RGGB2BGR_VNG = cv2.COLOR_BAYER_RGGB2BGR_VNG
    BayerGRBG2BGR_VNG = cv2.COLOR_BayerGRBG2BGR_VNG
    BAYER_GRBG2BGR_VNG = cv2.COLOR_BAYER_GRBG2BGR_VNG
    BayerBGGR2BGR_VNG = cv2.COLOR_BayerBGGR2BGR_VNG
    BAYER_BGGR2BGR_VNG = cv2.COLOR_BAYER_BGGR2BGR_VNG
    BayerGBRG2BGR_VNG = cv2.COLOR_BayerGBRG2BGR_VNG
    BAYER_GBRG2BGR_VNG = cv2.COLOR_BAYER_GBRG2BGR_VNG
    BayerRGGB2RGB_VNG = cv2.COLOR_BayerRGGB2RGB_VNG
    BAYER_RGGB2RGB_VNG = cv2.COLOR_BAYER_RGGB2RGB_VNG
    BayerGRBG2RGB_VNG = cv2.COLOR_BayerGRBG2RGB_VNG
    BAYER_GRBG2RGB_VNG = cv2.COLOR_BAYER_GRBG2RGB_VNG
    BayerBGGR2RGB_VNG = cv2.COLOR_BayerBGGR2RGB_VNG
    BAYER_BGGR2RGB_VNG = cv2.COLOR_BAYER_BGGR2RGB_VNG
    BayerGBRG2RGB_VNG = cv2.COLOR_BayerGBRG2RGB_VNG
    BAYER_GBRG2RGB_VNG = cv2.COLOR_BAYER_GBRG2RGB_VNG
    BayerBG2RGB_VNG = cv2.COLOR_BayerBG2RGB_VNG
    BAYER_BG2RGB_VNG = cv2.COLOR_BAYER_BG2RGB_VNG
    BayerGB2RGB_VNG = cv2.COLOR_BayerGB2RGB_VNG
    BAYER_GB2RGB_VNG = cv2.COLOR_BAYER_GB2RGB_VNG
    BayerRG2RGB_VNG = cv2.COLOR_BayerRG2RGB_VNG
    BAYER_RG2RGB_VNG = cv2.COLOR_BAYER_RG2RGB_VNG
    BayerGR2RGB_VNG = cv2.COLOR_BayerGR2RGB_VNG
    BAYER_GR2RGB_VNG = cv2.COLOR_BAYER_GR2RGB_VNG
    BayerBG2BGR_EA = cv2.COLOR_BayerBG2BGR_EA
    BAYER_BG2BGR_EA = cv2.COLOR_BAYER_BG2BGR_EA
    BayerGB2BGR_EA = cv2.COLOR_BayerGB2BGR_EA
    BAYER_GB2BGR_EA = cv2.COLOR_BAYER_GB2BGR_EA
    BayerRG2BGR_EA = cv2.COLOR_BayerRG2BGR_EA
    BAYER_RG2BGR_EA = cv2.COLOR_BAYER_RG2BGR_EA
    BayerGR2BGR_EA = cv2.COLOR_BayerGR2BGR_EA
    BAYER_GR2BGR_EA = cv2.COLOR_BAYER_GR2BGR_EA
    BayerRGGB2BGR_EA = cv2.COLOR_BayerRGGB2BGR_EA
    BAYER_RGGB2BGR_EA = cv2.COLOR_BAYER_RGGB2BGR_EA
    BayerGRBG2BGR_EA = cv2.COLOR_BayerGRBG2BGR_EA
    BAYER_GRBG2BGR_EA = cv2.COLOR_BAYER_GRBG2BGR_EA
    BayerBGGR2BGR_EA = cv2.COLOR_BayerBGGR2BGR_EA
    BAYER_BGGR2BGR_EA = cv2.COLOR_BAYER_BGGR2BGR_EA
    BayerGBRG2BGR_EA = cv2.COLOR_BayerGBRG2BGR_EA
    BAYER_GBRG2BGR_EA = cv2.COLOR_BAYER_GBRG2BGR_EA
    BayerRGGB2RGB_EA = cv2.COLOR_BayerRGGB2RGB_EA
    BAYER_RGGB2RGB_EA = cv2.COLOR_BAYER_RGGB2RGB_EA
    BayerGRBG2RGB_EA = cv2.COLOR_BayerGRBG2RGB_EA
    BAYER_GRBG2RGB_EA = cv2.COLOR_BAYER_GRBG2RGB_EA
    BayerBGGR2RGB_EA = cv2.COLOR_BayerBGGR2RGB_EA
    BAYER_BGGR2RGB_EA = cv2.COLOR_BAYER_BGGR2RGB_EA
    BayerGBRG2RGB_EA = cv2.COLOR_BayerGBRG2RGB_EA
    BAYER_GBRG2RGB_EA = cv2.COLOR_BAYER_GBRG2RGB_EA
    BayerBG2RGB_EA = cv2.COLOR_BayerBG2RGB_EA
    BAYER_BG2RGB_EA = cv2.COLOR_BAYER_BG2RGB_EA
    BayerGB2RGB_EA = cv2.COLOR_BayerGB2RGB_EA
    BAYER_GB2RGB_EA = cv2.COLOR_BAYER_GB2RGB_EA
    BayerRG2RGB_EA = cv2.COLOR_BayerRG2RGB_EA
    BAYER_RG2RGB_EA = cv2.COLOR_BAYER_RG2RGB_EA
    BayerGR2RGB_EA = cv2.COLOR_BayerGR2RGB_EA
    BAYER_GR2RGB_EA = cv2.COLOR_BAYER_GR2RGB_EA
    BayerBG2BGRA = cv2.COLOR_BayerBG2BGRA
    BAYER_BG2BGRA = cv2.COLOR_BAYER_BG2BGRA
    BayerGB2BGRA = cv2.COLOR_BayerGB2BGRA
    BAYER_GB2BGRA = cv2.COLOR_BAYER_GB2BGRA
    BayerRG2BGRA = cv2.COLOR_BayerRG2BGRA
    BAYER_RG2BGRA = cv2.COLOR_BAYER_RG2BGRA
    BayerGR2BGRA = cv2.COLOR_BayerGR2BGRA
    BAYER_GR2BGRA = cv2.COLOR_BAYER_GR2BGRA
    BayerRGGB2BGRA = cv2.COLOR_BayerRGGB2BGRA
    BAYER_RGGB2BGRA = cv2.COLOR_BAYER_RGGB2BGRA
    BayerGRBG2BGRA = cv2.COLOR_BayerGRBG2BGRA
    BAYER_GRBG2BGRA = cv2.COLOR_BAYER_GRBG2BGRA
    BayerBGGR2BGRA = cv2.COLOR_BayerBGGR2BGRA
    BAYER_BGGR2BGRA = cv2.COLOR_BAYER_BGGR2BGRA
    BayerGBRG2BGRA = cv2.COLOR_BayerGBRG2BGRA
    BAYER_GBRG2BGRA = cv2.COLOR_BAYER_GBRG2BGRA
    BayerRGGB2RGBA = cv2.COLOR_BayerRGGB2RGBA
    BAYER_RGGB2RGBA = cv2.COLOR_BAYER_RGGB2RGBA
    BayerGRBG2RGBA = cv2.COLOR_BayerGRBG2RGBA
    BAYER_GRBG2RGBA = cv2.COLOR_BAYER_GRBG2RGBA
    BayerBGGR2RGBA = cv2.COLOR_BayerBGGR2RGBA
    BAYER_BGGR2RGBA = cv2.COLOR_BAYER_BGGR2RGBA
    BayerGBRG2RGBA = cv2.COLOR_BayerGBRG2RGBA
    BAYER_GBRG2RGBA = cv2.COLOR_BAYER_GBRG2RGBA
    BayerBG2RGBA = cv2.COLOR_BayerBG2RGBA
    BAYER_BG2RGBA = cv2.COLOR_BAYER_BG2RGBA
    BayerGB2RGBA = cv2.COLOR_BayerGB2RGBA
    BAYER_GB2RGBA = cv2.COLOR_BAYER_GB2RGBA
    BayerRG2RGBA = cv2.COLOR_BayerRG2RGBA
    BAYER_RG2RGBA = cv2.COLOR_BAYER_RG2RGBA
    BayerGR2RGBA = cv2.COLOR_BayerGR2RGBA
    BAYER_GR2RGBA = cv2.COLOR_BAYER_GR2RGBA


def cvt_color(image: NDArray, code: CvtColorCode) -> NDArray:
    return cv2.cvtColor(image, code.value)


cvt_color_BGR2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BGR2BGRA)
cvt_color_RGB2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_RGB2RGBA)
cvt_color_BGRA2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BGRA2BGR)
cvt_color_RGBA2RGB = partial(cv2.cvtColor, code=cv2.COLOR_RGBA2RGB)
cvt_color_BGR2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGBA)
cvt_color_RGB2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_RGB2BGRA)
cvt_color_RGBA2BGR = partial(cv2.cvtColor, code=cv2.COLOR_RGBA2BGR)
cvt_color_BGRA2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BGRA2RGB)
cvt_color_BGR2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB)
cvt_color_RGB2BGR = partial(cv2.cvtColor, code=cv2.COLOR_RGB2BGR)
cvt_color_BGRA2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BGRA2RGBA)
cvt_color_RGBA2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_RGBA2BGRA)
cvt_color_BGR2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BGR2GRAY)
cvt_color_RGB2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_RGB2GRAY)
cvt_color_GRAY2BGR = partial(cv2.cvtColor, code=cv2.COLOR_GRAY2BGR)
cvt_color_GRAY2RGB = partial(cv2.cvtColor, code=cv2.COLOR_GRAY2RGB)
cvt_color_GRAY2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_GRAY2BGRA)
cvt_color_GRAY2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_GRAY2RGBA)
cvt_color_BGRA2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BGRA2GRAY)
cvt_color_RGBA2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_RGBA2GRAY)
cvt_color_BGR2BGR565 = partial(cv2.cvtColor, code=cv2.COLOR_BGR2BGR565)
cvt_color_RGB2BGR565 = partial(cv2.cvtColor, code=cv2.COLOR_RGB2BGR565)
cvt_color_BGR5652BGR = partial(cv2.cvtColor, code=cv2.COLOR_BGR5652BGR)
cvt_color_BGR5652RGB = partial(cv2.cvtColor, code=cv2.COLOR_BGR5652RGB)
cvt_color_BGRA2BGR565 = partial(cv2.cvtColor, code=cv2.COLOR_BGRA2BGR565)
cvt_color_RGBA2BGR565 = partial(cv2.cvtColor, code=cv2.COLOR_RGBA2BGR565)
cvt_color_BGR5652BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BGR5652BGRA)
cvt_color_BGR5652RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BGR5652RGBA)
cvt_color_GRAY2BGR565 = partial(cv2.cvtColor, code=cv2.COLOR_GRAY2BGR565)
cvt_color_BGR5652GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BGR5652GRAY)
cvt_color_BGR2BGR555 = partial(cv2.cvtColor, code=cv2.COLOR_BGR2BGR555)
cvt_color_RGB2BGR555 = partial(cv2.cvtColor, code=cv2.COLOR_RGB2BGR555)
cvt_color_BGR5552BGR = partial(cv2.cvtColor, code=cv2.COLOR_BGR5552BGR)
cvt_color_BGR5552RGB = partial(cv2.cvtColor, code=cv2.COLOR_BGR5552RGB)
cvt_color_BGRA2BGR555 = partial(cv2.cvtColor, code=cv2.COLOR_BGRA2BGR555)
cvt_color_RGBA2BGR555 = partial(cv2.cvtColor, code=cv2.COLOR_RGBA2BGR555)
cvt_color_BGR5552BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BGR5552BGRA)
cvt_color_BGR5552RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BGR5552RGBA)
cvt_color_GRAY2BGR555 = partial(cv2.cvtColor, code=cv2.COLOR_GRAY2BGR555)
cvt_color_BGR5552GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BGR5552GRAY)
cvt_color_BGR2XYZ = partial(cv2.cvtColor, code=cv2.COLOR_BGR2XYZ)
cvt_color_RGB2XYZ = partial(cv2.cvtColor, code=cv2.COLOR_RGB2XYZ)
cvt_color_XYZ2BGR = partial(cv2.cvtColor, code=cv2.COLOR_XYZ2BGR)
cvt_color_XYZ2RGB = partial(cv2.cvtColor, code=cv2.COLOR_XYZ2RGB)
cvt_color_BGR2YCrCb = partial(cv2.cvtColor, code=cv2.COLOR_BGR2YCrCb)
cvt_color_BGR2YCR_CB = partial(cv2.cvtColor, code=cv2.COLOR_BGR2YCR_CB)
cvt_color_RGB2YCrCb = partial(cv2.cvtColor, code=cv2.COLOR_RGB2YCrCb)
cvt_color_RGB2YCR_CB = partial(cv2.cvtColor, code=cv2.COLOR_RGB2YCR_CB)
cvt_color_YCrCb2BGR = partial(cv2.cvtColor, code=cv2.COLOR_YCrCb2BGR)
cvt_color_YCR_CB2BGR = partial(cv2.cvtColor, code=cv2.COLOR_YCR_CB2BGR)
cvt_color_YCrCb2RGB = partial(cv2.cvtColor, code=cv2.COLOR_YCrCb2RGB)
cvt_color_YCR_CB2RGB = partial(cv2.cvtColor, code=cv2.COLOR_YCR_CB2RGB)
cvt_color_BGR2HSV = partial(cv2.cvtColor, code=cv2.COLOR_BGR2HSV)
cvt_color_RGB2HSV = partial(cv2.cvtColor, code=cv2.COLOR_RGB2HSV)
cvt_color_BGR2Lab = partial(cv2.cvtColor, code=cv2.COLOR_BGR2Lab)
cvt_color_BGR2LAB = partial(cv2.cvtColor, code=cv2.COLOR_BGR2LAB)
cvt_color_RGB2Lab = partial(cv2.cvtColor, code=cv2.COLOR_RGB2Lab)
cvt_color_RGB2LAB = partial(cv2.cvtColor, code=cv2.COLOR_RGB2LAB)
cvt_color_BGR2Luv = partial(cv2.cvtColor, code=cv2.COLOR_BGR2Luv)
cvt_color_BGR2LUV = partial(cv2.cvtColor, code=cv2.COLOR_BGR2LUV)
cvt_color_RGB2Luv = partial(cv2.cvtColor, code=cv2.COLOR_RGB2Luv)
cvt_color_RGB2LUV = partial(cv2.cvtColor, code=cv2.COLOR_RGB2LUV)
cvt_color_BGR2HLS = partial(cv2.cvtColor, code=cv2.COLOR_BGR2HLS)
cvt_color_RGB2HLS = partial(cv2.cvtColor, code=cv2.COLOR_RGB2HLS)
cvt_color_HSV2BGR = partial(cv2.cvtColor, code=cv2.COLOR_HSV2BGR)
cvt_color_HSV2RGB = partial(cv2.cvtColor, code=cv2.COLOR_HSV2RGB)
cvt_color_Lab2BGR = partial(cv2.cvtColor, code=cv2.COLOR_Lab2BGR)
cvt_color_LAB2BGR = partial(cv2.cvtColor, code=cv2.COLOR_LAB2BGR)
cvt_color_Lab2RGB = partial(cv2.cvtColor, code=cv2.COLOR_Lab2RGB)
cvt_color_LAB2RGB = partial(cv2.cvtColor, code=cv2.COLOR_LAB2RGB)
cvt_color_Luv2BGR = partial(cv2.cvtColor, code=cv2.COLOR_Luv2BGR)
cvt_color_LUV2BGR = partial(cv2.cvtColor, code=cv2.COLOR_LUV2BGR)
cvt_color_Luv2RGB = partial(cv2.cvtColor, code=cv2.COLOR_Luv2RGB)
cvt_color_LUV2RGB = partial(cv2.cvtColor, code=cv2.COLOR_LUV2RGB)
cvt_color_HLS2BGR = partial(cv2.cvtColor, code=cv2.COLOR_HLS2BGR)
cvt_color_HLS2RGB = partial(cv2.cvtColor, code=cv2.COLOR_HLS2RGB)
cvt_color_BGR2HSV_FULL = partial(cv2.cvtColor, code=cv2.COLOR_BGR2HSV_FULL)
cvt_color_RGB2HSV_FULL = partial(cv2.cvtColor, code=cv2.COLOR_RGB2HSV_FULL)
cvt_color_BGR2HLS_FULL = partial(cv2.cvtColor, code=cv2.COLOR_BGR2HLS_FULL)
cvt_color_RGB2HLS_FULL = partial(cv2.cvtColor, code=cv2.COLOR_RGB2HLS_FULL)
cvt_color_HSV2BGR_FULL = partial(cv2.cvtColor, code=cv2.COLOR_HSV2BGR_FULL)
cvt_color_HSV2RGB_FULL = partial(cv2.cvtColor, code=cv2.COLOR_HSV2RGB_FULL)
cvt_color_HLS2BGR_FULL = partial(cv2.cvtColor, code=cv2.COLOR_HLS2BGR_FULL)
cvt_color_HLS2RGB_FULL = partial(cv2.cvtColor, code=cv2.COLOR_HLS2RGB_FULL)
cvt_color_LBGR2Lab = partial(cv2.cvtColor, code=cv2.COLOR_LBGR2Lab)
cvt_color_LBGR2LAB = partial(cv2.cvtColor, code=cv2.COLOR_LBGR2LAB)
cvt_color_LRGB2Lab = partial(cv2.cvtColor, code=cv2.COLOR_LRGB2Lab)
cvt_color_LRGB2LAB = partial(cv2.cvtColor, code=cv2.COLOR_LRGB2LAB)
cvt_color_LBGR2Luv = partial(cv2.cvtColor, code=cv2.COLOR_LBGR2Luv)
cvt_color_LBGR2LUV = partial(cv2.cvtColor, code=cv2.COLOR_LBGR2LUV)
cvt_color_LRGB2Luv = partial(cv2.cvtColor, code=cv2.COLOR_LRGB2Luv)
cvt_color_LRGB2LUV = partial(cv2.cvtColor, code=cv2.COLOR_LRGB2LUV)
cvt_color_Lab2LBGR = partial(cv2.cvtColor, code=cv2.COLOR_Lab2LBGR)
cvt_color_LAB2LBGR = partial(cv2.cvtColor, code=cv2.COLOR_LAB2LBGR)
cvt_color_Lab2LRGB = partial(cv2.cvtColor, code=cv2.COLOR_Lab2LRGB)
cvt_color_LAB2LRGB = partial(cv2.cvtColor, code=cv2.COLOR_LAB2LRGB)
cvt_color_Luv2LBGR = partial(cv2.cvtColor, code=cv2.COLOR_Luv2LBGR)
cvt_color_LUV2LBGR = partial(cv2.cvtColor, code=cv2.COLOR_LUV2LBGR)
cvt_color_Luv2LRGB = partial(cv2.cvtColor, code=cv2.COLOR_Luv2LRGB)
cvt_color_LUV2LRGB = partial(cv2.cvtColor, code=cv2.COLOR_LUV2LRGB)
cvt_color_BGR2YUV = partial(cv2.cvtColor, code=cv2.COLOR_BGR2YUV)
cvt_color_RGB2YUV = partial(cv2.cvtColor, code=cv2.COLOR_RGB2YUV)
cvt_color_YUV2BGR = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR)
cvt_color_YUV2RGB = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB)
cvt_color_YUV2RGB_NV12 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB_NV12)
cvt_color_YUV2BGR_NV12 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_NV12)
cvt_color_YUV2RGB_NV21 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB_NV21)
cvt_color_YUV2BGR_NV21 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_NV21)
cvt_color_YUV420sp2RGB = partial(cv2.cvtColor, code=cv2.COLOR_YUV420sp2RGB)
cvt_color_YUV420SP2RGB = partial(cv2.cvtColor, code=cv2.COLOR_YUV420SP2RGB)
cvt_color_YUV420sp2BGR = partial(cv2.cvtColor, code=cv2.COLOR_YUV420sp2BGR)
cvt_color_YUV420SP2BGR = partial(cv2.cvtColor, code=cv2.COLOR_YUV420SP2BGR)
cvt_color_YUV2RGBA_NV12 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGBA_NV12)
cvt_color_YUV2BGRA_NV12 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGRA_NV12)
cvt_color_YUV2RGBA_NV21 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGBA_NV21)
cvt_color_YUV2BGRA_NV21 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGRA_NV21)
cvt_color_YUV420sp2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_YUV420sp2RGBA)
cvt_color_YUV420SP2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_YUV420SP2RGBA)
cvt_color_YUV420sp2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_YUV420sp2BGRA)
cvt_color_YUV420SP2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_YUV420SP2BGRA)
cvt_color_YUV2RGB_YV12 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB_YV12)
cvt_color_YUV2BGR_YV12 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_YV12)
cvt_color_YUV2RGB_IYUV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB_IYUV)
cvt_color_YUV2BGR_IYUV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_IYUV)
cvt_color_YUV2RGB_I420 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB_I420)
cvt_color_YUV2BGR_I420 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_I420)
cvt_color_YUV420p2RGB = partial(cv2.cvtColor, code=cv2.COLOR_YUV420p2RGB)
cvt_color_YUV420P2RGB = partial(cv2.cvtColor, code=cv2.COLOR_YUV420P2RGB)
cvt_color_YUV420p2BGR = partial(cv2.cvtColor, code=cv2.COLOR_YUV420p2BGR)
cvt_color_YUV420P2BGR = partial(cv2.cvtColor, code=cv2.COLOR_YUV420P2BGR)
cvt_color_YUV2RGBA_YV12 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGBA_YV12)
cvt_color_YUV2BGRA_YV12 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGRA_YV12)
cvt_color_YUV2RGBA_IYUV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGBA_IYUV)
cvt_color_YUV2BGRA_IYUV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGRA_IYUV)
cvt_color_YUV2RGBA_I420 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGBA_I420)
cvt_color_YUV2BGRA_I420 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGRA_I420)
cvt_color_YUV420p2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_YUV420p2RGBA)
cvt_color_YUV420P2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_YUV420P2RGBA)
cvt_color_YUV420p2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_YUV420p2BGRA)
cvt_color_YUV420P2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_YUV420P2BGRA)
cvt_color_YUV2GRAY_420 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_420)
cvt_color_YUV2GRAY_NV21 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_NV21)
cvt_color_YUV2GRAY_NV12 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_NV12)
cvt_color_YUV2GRAY_YV12 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_YV12)
cvt_color_YUV2GRAY_IYUV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_IYUV)
cvt_color_YUV2GRAY_I420 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_I420)
cvt_color_YUV420sp2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_YUV420sp2GRAY)
cvt_color_YUV420SP2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_YUV420SP2GRAY)
cvt_color_YUV420p2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_YUV420p2GRAY)
cvt_color_YUV420P2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_YUV420P2GRAY)
cvt_color_YUV2RGB_UYVY = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB_UYVY)
cvt_color_YUV2BGR_UYVY = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_UYVY)
cvt_color_YUV2RGB_Y422 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB_Y422)
cvt_color_YUV2BGR_Y422 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_Y422)
cvt_color_YUV2RGB_UYNV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB_UYNV)
cvt_color_YUV2BGR_UYNV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_UYNV)
cvt_color_YUV2RGBA_UYVY = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGBA_UYVY)
cvt_color_YUV2BGRA_UYVY = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGRA_UYVY)
cvt_color_YUV2RGBA_Y422 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGBA_Y422)
cvt_color_YUV2BGRA_Y422 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGRA_Y422)
cvt_color_YUV2RGBA_UYNV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGBA_UYNV)
cvt_color_YUV2BGRA_UYNV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGRA_UYNV)
cvt_color_YUV2RGB_YUY2 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB_YUY2)
cvt_color_YUV2BGR_YUY2 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_YUY2)
cvt_color_YUV2RGB_YVYU = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB_YVYU)
cvt_color_YUV2BGR_YVYU = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_YVYU)
cvt_color_YUV2RGB_YUYV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB_YUYV)
cvt_color_YUV2BGR_YUYV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_YUYV)
cvt_color_YUV2RGB_YUNV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGB_YUNV)
cvt_color_YUV2BGR_YUNV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR_YUNV)
cvt_color_YUV2RGBA_YUY2 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGBA_YUY2)
cvt_color_YUV2BGRA_YUY2 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGRA_YUY2)
cvt_color_YUV2RGBA_YVYU = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGBA_YVYU)
cvt_color_YUV2BGRA_YVYU = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGRA_YVYU)
cvt_color_YUV2RGBA_YUYV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGBA_YUYV)
cvt_color_YUV2BGRA_YUYV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGRA_YUYV)
cvt_color_YUV2RGBA_YUNV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2RGBA_YUNV)
cvt_color_YUV2BGRA_YUNV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGRA_YUNV)
cvt_color_YUV2GRAY_UYVY = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_UYVY)
cvt_color_YUV2GRAY_YUY2 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_YUY2)
cvt_color_YUV2GRAY_Y422 = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_Y422)
cvt_color_YUV2GRAY_UYNV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_UYNV)
cvt_color_YUV2GRAY_YVYU = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_YVYU)
cvt_color_YUV2GRAY_YUYV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_YUYV)
cvt_color_YUV2GRAY_YUNV = partial(cv2.cvtColor, code=cv2.COLOR_YUV2GRAY_YUNV)
cvt_color_RGBA2mRGBA = partial(cv2.cvtColor, code=cv2.COLOR_RGBA2mRGBA)
cvt_color_RGBA2M_RGBA = partial(cv2.cvtColor, code=cv2.COLOR_RGBA2M_RGBA)
cvt_color_mRGBA2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_mRGBA2RGBA)
cvt_color_M_RGBA2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_M_RGBA2RGBA)
cvt_color_RGB2YUV_I420 = partial(cv2.cvtColor, code=cv2.COLOR_RGB2YUV_I420)
cvt_color_BGR2YUV_I420 = partial(cv2.cvtColor, code=cv2.COLOR_BGR2YUV_I420)
cvt_color_RGB2YUV_IYUV = partial(cv2.cvtColor, code=cv2.COLOR_RGB2YUV_IYUV)
cvt_color_BGR2YUV_IYUV = partial(cv2.cvtColor, code=cv2.COLOR_BGR2YUV_IYUV)
cvt_color_RGBA2YUV_I420 = partial(cv2.cvtColor, code=cv2.COLOR_RGBA2YUV_I420)
cvt_color_BGRA2YUV_I420 = partial(cv2.cvtColor, code=cv2.COLOR_BGRA2YUV_I420)
cvt_color_RGBA2YUV_IYUV = partial(cv2.cvtColor, code=cv2.COLOR_RGBA2YUV_IYUV)
cvt_color_BGRA2YUV_IYUV = partial(cv2.cvtColor, code=cv2.COLOR_BGRA2YUV_IYUV)
cvt_color_RGB2YUV_YV12 = partial(cv2.cvtColor, code=cv2.COLOR_RGB2YUV_YV12)
cvt_color_BGR2YUV_YV12 = partial(cv2.cvtColor, code=cv2.COLOR_BGR2YUV_YV12)
cvt_color_RGBA2YUV_YV12 = partial(cv2.cvtColor, code=cv2.COLOR_RGBA2YUV_YV12)
cvt_color_BGRA2YUV_YV12 = partial(cv2.cvtColor, code=cv2.COLOR_BGRA2YUV_YV12)
cvt_color_BayerBG2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BayerBG2BGR)
cvt_color_BAYER_BG2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BG2BGR)
cvt_color_BayerGB2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BayerGB2BGR)
cvt_color_BAYER_GB2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GB2BGR)
cvt_color_BayerRG2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BayerRG2BGR)
cvt_color_BAYER_RG2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RG2BGR)
cvt_color_BayerGR2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BayerGR2BGR)
cvt_color_BAYER_GR2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GR2BGR)
cvt_color_BayerRGGB2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BayerRGGB2BGR)
cvt_color_BAYER_RGGB2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RGGB2BGR)
cvt_color_BayerGRBG2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BayerGRBG2BGR)
cvt_color_BAYER_GRBG2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GRBG2BGR)
cvt_color_BayerBGGR2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BayerBGGR2BGR)
cvt_color_BAYER_BGGR2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BGGR2BGR)
cvt_color_BayerGBRG2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BayerGBRG2BGR)
cvt_color_BAYER_GBRG2BGR = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GBRG2BGR)
cvt_color_BayerRGGB2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BayerRGGB2RGB)
cvt_color_BAYER_RGGB2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RGGB2RGB)
cvt_color_BayerGRBG2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BayerGRBG2RGB)
cvt_color_BAYER_GRBG2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GRBG2RGB)
cvt_color_BayerBGGR2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BayerBGGR2RGB)
cvt_color_BAYER_BGGR2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BGGR2RGB)
cvt_color_BayerGBRG2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BayerGBRG2RGB)
cvt_color_BAYER_GBRG2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GBRG2RGB)
cvt_color_BayerBG2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BayerBG2RGB)
cvt_color_BAYER_BG2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BG2RGB)
cvt_color_BayerGB2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BayerGB2RGB)
cvt_color_BAYER_GB2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GB2RGB)
cvt_color_BayerRG2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BayerRG2RGB)
cvt_color_BAYER_RG2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RG2RGB)
cvt_color_BayerGR2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BayerGR2RGB)
cvt_color_BAYER_GR2RGB = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GR2RGB)
cvt_color_BayerBG2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BayerBG2GRAY)
cvt_color_BAYER_BG2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BG2GRAY)
cvt_color_BayerGB2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BayerGB2GRAY)
cvt_color_BAYER_GB2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GB2GRAY)
cvt_color_BayerRG2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BayerRG2GRAY)
cvt_color_BAYER_RG2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RG2GRAY)
cvt_color_BayerGR2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BayerGR2GRAY)
cvt_color_BAYER_GR2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GR2GRAY)
cvt_color_BayerRGGB2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BayerRGGB2GRAY)
cvt_color_BAYER_RGGB2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RGGB2GRAY)
cvt_color_BayerGRBG2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BayerGRBG2GRAY)
cvt_color_BAYER_GRBG2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GRBG2GRAY)
cvt_color_BayerBGGR2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BayerBGGR2GRAY)
cvt_color_BAYER_BGGR2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BGGR2GRAY)
cvt_color_BayerGBRG2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BayerGBRG2GRAY)
cvt_color_BAYER_GBRG2GRAY = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GBRG2GRAY)
cvt_color_BayerBG2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerBG2BGR_VNG)
cvt_color_BAYER_BG2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BG2BGR_VNG)
cvt_color_BayerGB2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerGB2BGR_VNG)
cvt_color_BAYER_GB2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GB2BGR_VNG)
cvt_color_BayerRG2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerRG2BGR_VNG)
cvt_color_BAYER_RG2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RG2BGR_VNG)
cvt_color_BayerGR2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerGR2BGR_VNG)
cvt_color_BAYER_GR2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GR2BGR_VNG)
cvt_color_BayerRGGB2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerRGGB2BGR_VNG)
cvt_color_BAYER_RGGB2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RGGB2BGR_VNG)
cvt_color_BayerGRBG2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerGRBG2BGR_VNG)
cvt_color_BAYER_GRBG2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GRBG2BGR_VNG)
cvt_color_BayerBGGR2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerBGGR2BGR_VNG)
cvt_color_BAYER_BGGR2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BGGR2BGR_VNG)
cvt_color_BayerGBRG2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerGBRG2BGR_VNG)
cvt_color_BAYER_GBRG2BGR_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GBRG2BGR_VNG)
cvt_color_BayerRGGB2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerRGGB2RGB_VNG)
cvt_color_BAYER_RGGB2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RGGB2RGB_VNG)
cvt_color_BayerGRBG2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerGRBG2RGB_VNG)
cvt_color_BAYER_GRBG2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GRBG2RGB_VNG)
cvt_color_BayerBGGR2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerBGGR2RGB_VNG)
cvt_color_BAYER_BGGR2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BGGR2RGB_VNG)
cvt_color_BayerGBRG2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerGBRG2RGB_VNG)
cvt_color_BAYER_GBRG2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GBRG2RGB_VNG)
cvt_color_BayerBG2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerBG2RGB_VNG)
cvt_color_BAYER_BG2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BG2RGB_VNG)
cvt_color_BayerGB2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerGB2RGB_VNG)
cvt_color_BAYER_GB2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GB2RGB_VNG)
cvt_color_BayerRG2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerRG2RGB_VNG)
cvt_color_BAYER_RG2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RG2RGB_VNG)
cvt_color_BayerGR2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BayerGR2RGB_VNG)
cvt_color_BAYER_GR2RGB_VNG = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GR2RGB_VNG)
cvt_color_BayerBG2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerBG2BGR_EA)
cvt_color_BAYER_BG2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BG2BGR_EA)
cvt_color_BayerGB2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGB2BGR_EA)
cvt_color_BAYER_GB2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GB2BGR_EA)
cvt_color_BayerRG2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerRG2BGR_EA)
cvt_color_BAYER_RG2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RG2BGR_EA)
cvt_color_BayerGR2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGR2BGR_EA)
cvt_color_BAYER_GR2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GR2BGR_EA)
cvt_color_BayerRGGB2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerRGGB2BGR_EA)
cvt_color_BAYER_RGGB2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RGGB2BGR_EA)
cvt_color_BayerGRBG2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGRBG2BGR_EA)
cvt_color_BAYER_GRBG2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GRBG2BGR_EA)
cvt_color_BayerBGGR2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerBGGR2BGR_EA)
cvt_color_BAYER_BGGR2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BGGR2BGR_EA)
cvt_color_BayerGBRG2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGBRG2BGR_EA)
cvt_color_BAYER_GBRG2BGR_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GBRG2BGR_EA)
cvt_color_BayerRGGB2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerRGGB2RGB_EA)
cvt_color_BAYER_RGGB2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RGGB2RGB_EA)
cvt_color_BayerGRBG2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGRBG2RGB_EA)
cvt_color_BAYER_GRBG2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GRBG2RGB_EA)
cvt_color_BayerBGGR2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerBGGR2RGB_EA)
cvt_color_BAYER_BGGR2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BGGR2RGB_EA)
cvt_color_BayerGBRG2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGBRG2RGB_EA)
cvt_color_BAYER_GBRG2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GBRG2RGB_EA)
cvt_color_BayerBG2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerBG2RGB_EA)
cvt_color_BAYER_BG2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BG2RGB_EA)
cvt_color_BayerGB2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGB2RGB_EA)
cvt_color_BAYER_GB2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GB2RGB_EA)
cvt_color_BayerRG2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerRG2RGB_EA)
cvt_color_BAYER_RG2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RG2RGB_EA)
cvt_color_BayerGR2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGR2RGB_EA)
cvt_color_BAYER_GR2RGB_EA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GR2RGB_EA)
cvt_color_BayerBG2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BayerBG2BGRA)
cvt_color_BAYER_BG2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BG2BGRA)
cvt_color_BayerGB2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGB2BGRA)
cvt_color_BAYER_GB2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GB2BGRA)
cvt_color_BayerRG2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BayerRG2BGRA)
cvt_color_BAYER_RG2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RG2BGRA)
cvt_color_BayerGR2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGR2BGRA)
cvt_color_BAYER_GR2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GR2BGRA)
cvt_color_BayerRGGB2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BayerRGGB2BGRA)
cvt_color_BAYER_RGGB2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RGGB2BGRA)
cvt_color_BayerGRBG2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGRBG2BGRA)
cvt_color_BAYER_GRBG2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GRBG2BGRA)
cvt_color_BayerBGGR2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BayerBGGR2BGRA)
cvt_color_BAYER_BGGR2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BGGR2BGRA)
cvt_color_BayerGBRG2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGBRG2BGRA)
cvt_color_BAYER_GBRG2BGRA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GBRG2BGRA)
cvt_color_BayerRGGB2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BayerRGGB2RGBA)
cvt_color_BAYER_RGGB2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RGGB2RGBA)
cvt_color_BayerGRBG2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGRBG2RGBA)
cvt_color_BAYER_GRBG2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GRBG2RGBA)
cvt_color_BayerBGGR2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BayerBGGR2RGBA)
cvt_color_BAYER_BGGR2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BGGR2RGBA)
cvt_color_BayerGBRG2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGBRG2RGBA)
cvt_color_BAYER_GBRG2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GBRG2RGBA)
cvt_color_BayerBG2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BayerBG2RGBA)
cvt_color_BAYER_BG2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_BG2RGBA)
cvt_color_BayerGB2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGB2RGBA)
cvt_color_BAYER_GB2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GB2RGBA)
cvt_color_BayerRG2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BayerRG2RGBA)
cvt_color_BAYER_RG2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_RG2RGBA)
cvt_color_BayerGR2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BayerGR2RGBA)
cvt_color_BAYER_GR2RGBA = partial(cv2.cvtColor, code=cv2.COLOR_BAYER_GR2RGBA)


class CvlCvtColor:
    @staticmethod
    def cvl_cvt_color(image: NDArray, code: CvtColorCode):
        return cvt_color(image, code)

    @staticmethod
    def cvl_cvt_color_bgr2rgb(image: NDArray):
        return cvt_color(image, CvtColorCode.BGR2RGB)

    @staticmethod
    def cvl_cvt_color_rgb2bgr(image: NDArray):
        return cvt_color(image, CvtColorCode.RGB2BGR)

    @staticmethod
    def cvl_cvt_color_rgba2bgr(image: NDArray):
        return cvt_color(image, CvtColorCode.RGBA2BGR)

    @staticmethod
    def cvl_cvt_color_bgr2gray(image: NDArray):
        return cvt_color(image, CvtColorCode.BGR2GRAY)

    @staticmethod
    def cvl_cvt_color_gray2bgr(image: NDArray):
        return cvt_color(image, CvtColorCode.GRAY2BGR)

    @staticmethod
    def cvl_cvt_color_bgr2hsv(image: NDArray):
        return cvt_color(image, CvtColorCode.BGR2HSV)

    @staticmethod
    def cvl_cvt_color_hsv2bgr(image: NDArray):
        return cvt_color(image, CvtColorCode.HSV2BGR)

    @staticmethod
    def cvl_cvt_color_bgr2hls(image: NDArray):
        return cvt_color(image, CvtColorCode.BGR2HLS)

    @staticmethod
    def cvl_cvt_color_hls2bgr(image: NDArray):
        return cvt_color(image, CvtColorCode.HLS2BGR)

    @staticmethod
    def cvl_cvt_color_bgr2yuv(image: NDArray):
        return cvt_color(image, CvtColorCode.BGR2YUV)

    @staticmethod
    def cvl_cvt_color_yuv2bgr(image: NDArray):
        return cvt_color(image, CvtColorCode.YUV2BGR)

    @staticmethod
    def cvl_cvt_color_bgr2ycrcb(image: NDArray):
        return cvt_color(image, CvtColorCode.BGR2YCR_CB)

    @staticmethod
    def cvl_cvt_color_ycrcb2bgr(image: NDArray):
        return cvt_color(image, CvtColorCode.YCR_CB2BGR)

    @staticmethod
    def cvl_cvt_color_bgr2lab(image: NDArray):
        return cvt_color(image, CvtColorCode.BGR2LAB)

    @staticmethod
    def cvl_cvt_color_lab2bgr(image: NDArray):
        return cvt_color(image, CvtColorCode.LAB2BGR)

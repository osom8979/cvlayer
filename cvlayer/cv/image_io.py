# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final, Optional, Sequence

import cv2
from numpy.typing import NDArray

WINDOWS_BITMAP_SUFFIX: Final[Sequence[str]] = ".bmp", ".dib"
JPEG_SUFFIX: Final[Sequence[str]] = ".jpeg", ".jpg", ".jpe"
JPEG_2000_SUFFIX: Final[Sequence[str]] = (".jp2",)
PNG_SUFFIX: Final[Sequence[str]] = (".png",)
WEBP_SUFFIX: Final[Sequence[str]] = (".webp",)
PORTABLE_IMAGE_SUFFIX: Final[Sequence[str]] = ".pbm", ".pgm", ".ppm", ".pxm", ".pnm"
SUN_RASTER_SUFFIX: Final[Sequence[str]] = ".sr", ".ras"
TIFF_SUFFIX: Final[Sequence[str]] = ".tiff", ".tif"
OPENEXR_SUFFIX: Final[Sequence[str]] = (".exr",)
RADIANCE_HDR_SUFFIX: Final[Sequence[str]] = ".hdr", ".pic"


@unique
class ImWriteJpegSamplingFactor(Enum):
    F411 = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_411
    F420 = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420
    F422 = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422
    F440 = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_440
    F444 = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444


@unique
class ImWritePngStrategy(Enum):
    DEFAULT = cv2.IMWRITE_PNG_STRATEGY_DEFAULT
    FILTERED = cv2.IMWRITE_PNG_STRATEGY_FILTERED
    HUFFMAN_ONLY = cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY
    RLE = cv2.IMWRITE_PNG_STRATEGY_RLE
    FIXED = cv2.IMWRITE_PNG_STRATEGY_FIXED


@unique
class ImWriteFlags(Enum):
    JPEG_QUALITY = cv2.IMWRITE_JPEG_QUALITY
    """
    For JPEG, it can be a quality from 0 to 100 (the higher is the better).
    Default value is 95.
    """

    JPEG_PROGRESSIVE = cv2.IMWRITE_JPEG_PROGRESSIVE
    """
    Enable JPEG features, 0 or 1, default is False.
    """

    JPEG_OPTIMIZE = cv2.IMWRITE_JPEG_OPTIMIZE
    """
    Enable JPEG features, 0 or 1, default is False.
    """

    JPEG_RST_INTERVAL = cv2.IMWRITE_JPEG_RST_INTERVAL
    """
    JPEG restart interval, 0 - 65535, default is 0 - no restart.
    """

    JPEG_LUMA_QUALITY = cv2.IMWRITE_JPEG_LUMA_QUALITY
    """
    Separate luma quality level, 0 - 100, default is -1 - don't use.
    """

    JPEG_CHROMA_QUALITY = cv2.IMWRITE_JPEG_CHROMA_QUALITY
    """
    Separate chroma quality level, 0 - 100, default is -1 - don't use.
    """

    JPEG_SAMPLING_FACTOR = cv2.IMWRITE_JPEG_SAMPLING_FACTOR
    """
    For JPEG, set sampling factor. See cv2.ImwriteJPEGSamplingFactorParams.
    """

    PNG_COMPRESSION = cv2.IMWRITE_PNG_COMPRESSION
    """
    For PNG, it can be the compression level from 0 to 9.
    A higher value means a smaller size and longer compression time.
    If specified,
    strategy is changed to IMWRITE_PNG_STRATEGY_DEFAULT (Z_DEFAULT_STRATEGY).
    Default value is 1 (best speed setting).
    """

    PNG_STRATEGY = cv2.IMWRITE_PNG_STRATEGY
    """
    One of cv2.ImwritePNGFlags, default is IMWRITE_PNG_STRATEGY_RLE.
    """

    PNG_BILEVEL = cv2.IMWRITE_PNG_BILEVEL
    """
    Binary level PNG, 0 or 1, default is 0.
    """

    PXM_BINARY = cv2.IMWRITE_PXM_BINARY
    """
    For PPM, PGM, or PBM, it can be a binary format flag, 0 or 1. Default value is 1.
    """

    EXR_TYPE = cv2.IMWRITE_EXR_TYPE
    WEBP_QUALITY = cv2.IMWRITE_WEBP_QUALITY
    """
    Override EXR storage type (FLOAT (FP32) is default)

    For WEBP, it can be a quality from 1 to 100 (the higher is the better).
    By default (without any parameter) and for quality above 100 the lossless
    compression is used.
    """

    HDR_COMPRESSION = cv2.IMWRITE_HDR_COMPRESSION
    PAM_TUPLETYPE = cv2.IMWRITE_PAM_TUPLETYPE
    """
    specify HDR compression

    For PAM, sets the TUPLETYPE field to the corresponding string value that is defined
    for the format
    """

    TIFF_RESUNIT = cv2.IMWRITE_TIFF_RESUNIT
    """
    For TIFF, use to specify which DPI resolution unit to set;
    see libtiff documentation for valid values.
    """

    TIFF_XDPI = cv2.IMWRITE_TIFF_XDPI
    """
    For TIFF, use to specify the X direction DPI.
    """

    TIFF_YDPI = cv2.IMWRITE_TIFF_YDPI
    """
    For TIFF, use to specify the Y direction DPI.
    """

    TIFF_COMPRESSION = cv2.IMWRITE_TIFF_COMPRESSION
    """
    For TIFF, use to specify the image compression scheme.
    See libtiff for integer constants corresponding to compression formats.

    Note,
    for images whose depth is CV_32F, only libtiff's SGILOG compression scheme is used.
    For other supported depths, the compression scheme can be specified by this flag;
    LZW compression is the default.
    """


@unique
class ImReadModes(Enum):
    UNCHANGED = cv2.IMREAD_UNCHANGED
    """
    If set, return the loaded image as is
    (with alpha channel, otherwise it gets cropped).
    Ignore EXIF orientation.
    """

    GRAYSCALE = cv2.IMREAD_GRAYSCALE
    """
    If set, always convert image to the single channel grayscale image
    (codec internal conversion).
    """

    COLOR = cv2.IMREAD_COLOR
    """
    If set, always convert image to the 3 channel BGR color image.
    """

    ANYDEPTH = cv2.IMREAD_ANYDEPTH
    """
    If set, return 16-bit/32-bit image when the input has the corresponding depth,
    otherwise convert it to 8-bit.
    """

    ANYCOLOR = cv2.IMREAD_ANYCOLOR
    """
    If set, the image is read in any possible color format.
    """

    LOAD_GDAL = cv2.IMREAD_LOAD_GDAL
    """
    If set, use the gdal driver for loading the image.
    """

    REDUCED_GRAYSCALE_2 = cv2.IMREAD_REDUCED_GRAYSCALE_2
    """
    If set, always convert image to the single channel grayscale image and
    the image size reduced 1/2.
    """

    REDUCED_COLOR_2 = cv2.IMREAD_REDUCED_COLOR_2
    """
    If set, always convert image to the 3 channel BGR color image and
    the image size reduced 1/2.
    """

    REDUCED_GRAYSCALE_4 = cv2.IMREAD_REDUCED_GRAYSCALE_4
    """
    If set, always convert image to the single channel grayscale image and
    the image size reduced 1/4.
    """

    REDUCED_COLOR_4 = cv2.IMREAD_REDUCED_COLOR_4
    """
    If set, always convert image to the 3 channel BGR color image and the
    image size reduced 1/4.
    """

    REDUCED_GRAYSCALE_8 = cv2.IMREAD_REDUCED_GRAYSCALE_8
    """
    If set, always convert image to the single channel grayscale image and
    the image size reduced 1/8.
    """

    REDUCED_COLOR_8 = cv2.IMREAD_REDUCED_COLOR_8
    """
    If set, always convert image to the 3 channel BGR color image and
    the image size reduced 1/8.
    """

    IGNORE_ORIENTATION = cv2.IMREAD_IGNORE_ORIENTATION
    """
    If set, do not rotate the image according to EXIF's orientation flag.
    """


def image_write(
    filename: str,
    image: NDArray,
    params: Optional[Sequence[int]] = None,
) -> bool:
    if params:
        return cv2.imwrite(filename, image, params)
    else:
        return cv2.imwrite(filename, image)


def image_write_jpeg(
    filename: str,
    image: NDArray,
    *,
    quality: Optional[int] = None,
    progressive: Optional[bool] = None,
    optimize: Optional[bool] = None,
    rst_interval: Optional[int] = None,
    luma_quality: Optional[int] = None,
    chroma_quality: Optional[int] = None,
    sampling_factor: Optional[ImWriteJpegSamplingFactor] = None,
) -> bool:
    assert any([filename.lower().endswith(suffix) for suffix in JPEG_SUFFIX])
    params = list()
    if quality is not None:
        assert 0 <= quality <= 100
        params += [ImWriteFlags.JPEG_QUALITY.value, quality]
    if progressive is not None:
        params += [ImWriteFlags.JPEG_PROGRESSIVE.value, 1 if progressive else 0]
    if optimize is not None:
        params += [ImWriteFlags.JPEG_OPTIMIZE.value, 1 if optimize else 0]
    if rst_interval is not None:
        assert 0 <= rst_interval <= 65535
        params += [ImWriteFlags.JPEG_RST_INTERVAL.value, rst_interval]
    if luma_quality is not None:
        assert 0 <= luma_quality <= 100 or luma_quality == -1
        params += [ImWriteFlags.JPEG_LUMA_QUALITY.value, luma_quality]
    if chroma_quality is not None:
        assert 0 <= chroma_quality <= 100 or chroma_quality == -1
        params += [ImWriteFlags.JPEG_CHROMA_QUALITY.value, chroma_quality]
    if sampling_factor is not None:
        params += [ImWriteFlags.JPEG_SAMPLING_FACTOR.value, sampling_factor.value]
    return image_write(filename, image, params)


def image_write_png(
    filename: str,
    image: NDArray,
    *,
    compression: Optional[int] = None,
    strategy: Optional[ImWritePngStrategy] = None,
    binary_level: Optional[bool] = None,
) -> bool:
    assert any([filename.lower().endswith(suffix) for suffix in PNG_SUFFIX])
    params = list()
    if compression is not None:
        assert 0 <= compression <= 9
        params += [ImWriteFlags.PNG_COMPRESSION.value, compression]
    if strategy is not None:
        params += [ImWriteFlags.PNG_STRATEGY.value, strategy.value]
    if binary_level is not None:
        params += [ImWriteFlags.PNG_BILEVEL.value, 1 if binary_level else 0]
    return image_write(filename, image, params)


def image_read(filename: str, flags: Optional[int] = None) -> NDArray:
    if flags is not None:
        return cv2.imread(filename, flags)
    else:
        return cv2.imread(filename)


def imread_modes_as_flags(modes: Sequence[ImReadModes]) -> int:
    result = 0
    for m in modes:
        result |= int(m.value)
    return result


def image_read_with_modes(filename: str, modes: Sequence[ImReadModes]) -> NDArray:
    return image_read(filename, imread_modes_as_flags(modes))


def image_read_with_kwargs(
    filename: str,
    *,
    unchanged=False,
    grayscale=False,
    color=False,
    anydepth=False,
    anycolor=False,
    load_gdal=False,
    reduced_grayscale_2=False,
    reduced_color_2=False,
    reduced_grayscale_4=False,
    reduced_color_4=False,
    reduced_grayscale_8=False,
    reduced_color_8=False,
    ignore_orientation=False,
) -> NDArray:
    modes = list()
    if unchanged:
        modes.append(ImReadModes.UNCHANGED)
    if grayscale:
        modes.append(ImReadModes.GRAYSCALE)
    if color:
        modes.append(ImReadModes.COLOR)
    if anydepth:
        modes.append(ImReadModes.ANYDEPTH)
    if anycolor:
        modes.append(ImReadModes.ANYCOLOR)
    if load_gdal:
        modes.append(ImReadModes.LOAD_GDAL)
    if reduced_grayscale_2:
        modes.append(ImReadModes.REDUCED_GRAYSCALE_2)
    if reduced_color_2:
        modes.append(ImReadModes.REDUCED_COLOR_2)
    if reduced_grayscale_4:
        modes.append(ImReadModes.REDUCED_GRAYSCALE_4)
    if reduced_color_4:
        modes.append(ImReadModes.REDUCED_COLOR_4)
    if reduced_grayscale_8:
        modes.append(ImReadModes.REDUCED_GRAYSCALE_8)
    if reduced_color_8:
        modes.append(ImReadModes.REDUCED_COLOR_8)
    if ignore_orientation:
        modes.append(ImReadModes.IGNORE_ORIENTATION)
    return image_read_with_modes(filename, modes)


class CvlImageIo:
    @staticmethod
    def cvl_image_read(filename: str, flags: int) -> NDArray:
        return image_read(filename, flags)

    @staticmethod
    def cvl_image_write_jpeg(
        filename: str,
        image: NDArray,
        *,
        quality: Optional[int] = None,
        progressive: Optional[bool] = None,
        optimize: Optional[bool] = None,
        rst_interval: Optional[int] = None,
        luma_quality: Optional[int] = None,
        chroma_quality: Optional[int] = None,
        sampling_factor: Optional[ImWriteJpegSamplingFactor] = None,
    ) -> bool:
        return image_write_jpeg(
            filename,
            image,
            quality=quality,
            progressive=progressive,
            optimize=optimize,
            rst_interval=rst_interval,
            luma_quality=luma_quality,
            chroma_quality=chroma_quality,
            sampling_factor=sampling_factor,
        )

    @staticmethod
    def cvl_image_write_png(
        filename: str,
        image: NDArray,
        *,
        compression: Optional[int] = None,
        strategy: Optional[ImWritePngStrategy] = None,
        binary_level: Optional[bool] = None,
    ) -> bool:
        return image_write_png(
            filename,
            image,
            compression=compression,
            strategy=strategy,
            binary_level=binary_level,
        )

    @staticmethod
    def cvl_image_write(
        filename: str,
        image: NDArray,
        params: Optional[Sequence[int]] = None,
    ) -> bool:
        return image_write(filename, image, params)

    @staticmethod
    def cvl_imread_modes_as_flags(modes: Sequence[ImReadModes]) -> int:
        return imread_modes_as_flags(modes)

    @staticmethod
    def cvl_image_read_with_modes(
        filename: str,
        modes: Sequence[ImReadModes],
    ) -> NDArray:
        return image_read_with_modes(filename, modes)

    @staticmethod
    def cvl_image_read_with_kwargs(
        filename: str,
        *,
        unchanged=False,
        grayscale=False,
        color=False,
        anydepth=False,
        anycolor=False,
        load_gdal=False,
        reduced_grayscale_2=False,
        reduced_color_2=False,
        reduced_grayscale_4=False,
        reduced_color_4=False,
        reduced_grayscale_8=False,
        reduced_color_8=False,
        ignore_orientation=False,
    ) -> NDArray:
        return image_read_with_kwargs(
            filename,
            unchanged=unchanged,
            grayscale=grayscale,
            color=color,
            anydepth=anydepth,
            anycolor=anycolor,
            load_gdal=load_gdal,
            reduced_grayscale_2=reduced_grayscale_2,
            reduced_color_2=reduced_color_2,
            reduced_grayscale_4=reduced_grayscale_4,
            reduced_color_4=reduced_color_4,
            reduced_grayscale_8=reduced_grayscale_8,
            reduced_color_8=reduced_color_8,
            ignore_orientation=ignore_orientation,
        )

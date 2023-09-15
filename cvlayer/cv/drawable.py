# -*- coding: utf-8 -*-

from enum import Enum, unique
from typing import Final, List, Tuple

import cv2
from numpy import full, uint8

from cvlayer.typing import Color, Image, Number, PointT, RectInt, RectT, SizeInt

FILLED: Final[int] = cv2.FILLED

LINE_4: Final[int] = cv2.LINE_4  # Bresenham 4 Connect
LINE_8: Final[int] = cv2.LINE_8  # Bresenham 8 Connect
LINE_AA: Final[int] = cv2.LINE_AA  # Anti-Aliasing

MARKER_SIZE: Final[int] = 20

ARROWED_LINE_TIP_LENGTH: Final[float] = 0.1

FONT_HERSHEY_SIMPLEX: Final[int] = cv2.FONT_HERSHEY_SIMPLEX
FONT_HERSHEY_PLAIN: Final[int] = cv2.FONT_HERSHEY_PLAIN
FONT_HERSHEY_DUPLEX: Final[int] = cv2.FONT_HERSHEY_DUPLEX
FONT_HERSHEY_COMPLEX: Final[int] = cv2.FONT_HERSHEY_COMPLEX
FONT_HERSHEY_TRIPLEX: Final[int] = cv2.FONT_HERSHEY_TRIPLEX
FONT_HERSHEY_COMPLEX_SMALL: Final[int] = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONT_HERSHEY_SCRIPT_SIMPLEX: Final[int] = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
FONT_HERSHEY_SCRIPT_COMPLEX: Final[int] = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
FONT_ITALIC: Final[int] = cv2.FONT_ITALIC

RADIUS: Final[int] = 4
THICKNESS: Final[int] = 2
COLOR: Final[Color] = (0, 0, 0)
LINE_TYPE: Final[int] = LINE_AA

FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: Final[float] = 1.0

OUTLINE_FILL_COLOR: Final[Color] = (255, 255, 255)
OUTLINE_COLOR: Final[Color] = (0, 0, 0)
OUTLINE_THICKNESS: Final[int] = 9

MULTILINE_COLOR: Final[Color] = (220, 220, 220)
MULTILINE_LINEFEED: Final[str] = "\n"
MULTILINE_LINE_SPACING: Final[int] = 4
MULTILINE_BACKGROUND_COLOR: Final[Color] = (0, 0, 0)
MULTILINE_BACKGROUND_ALPHA: Final[float] = 0.4
MULTILINE_BOX_MARGIN: Final[int] = 8
MULTILINE_BOX_ANCHOR_X: Final[float] = 0.0
MULTILINE_BOX_ANCHOR_Y: Final[float] = 0.0

CROSSHAIR_POINT_RADIUS: Final[int] = 6
CROSSHAIR_POINT_THICKNESS: Final[int] = 1
CROSSHAIR_POINT_COLOR: Final[Color] = (0, 0, 255)
CROSSHAIR_POINT_LINE_TYPE: Final[int] = LINE_AA
CROSSHAIR_POINT_PADDING: Final[int] = 2


@unique
class DrawTextOrigin(Enum):
    BOTTOM_LEFT = True
    TOP_LEFT = False


@unique
class MarkerType(Enum):
    CROSS = cv2.MARKER_CROSS
    TILTED_CROSS = cv2.MARKER_TILTED_CROSS
    STAR = cv2.MARKER_STAR
    DIAMOND = cv2.MARKER_DIAMOND
    SQUARE = cv2.MARKER_SQUARE
    TRIANGLE_UP = cv2.MARKER_TRIANGLE_UP
    TRIANGLE_DOWN = cv2.MARKER_TRIANGLE_DOWN


def draw_point(
    image: Image,
    x: Number,
    y: Number,
    radius=RADIUS,
    color=COLOR,
    thickness=FILLED,
    line_type=LINE_TYPE,
) -> None:
    center = int(x), int(y)
    cv2.circle(image, center, radius, color, thickness, line_type)


def draw_line(
    image: Image,
    point1: PointT,
    point2: PointT,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
) -> None:
    x1, y1 = int(point1[0]), int(point1[1])
    x2, y2 = int(point2[0]), int(point2[1])
    cv2.line(image, (x1, y1), (x2, y2), color, thickness, line_type)


def draw_rectangle(
    image: Image,
    roi: RectT,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
) -> None:
    point1 = int(roi[0]), int(roi[1])
    point2 = int(roi[2]), int(roi[3])
    cv2.rectangle(image, point1, point2, color, thickness, line_type)


def draw_circle(
    image: Image,
    x: Number,
    y: Number,
    radius=RADIUS,
    color=COLOR,
    thickness=FILLED,
    line_type=LINE_TYPE,
) -> None:
    center = int(x), int(y)
    cv2.circle(image, center, radius, color, thickness, line_type)


def draw_ellipse(
    image: Image,
    center_x: Number,
    center_y: Number,
    axes_x: Number,
    axes_y: Number,
    angle: float,
    start_angle: float,
    end_angle: float,
    color=COLOR,
    thickness=FILLED,
    line_type=LINE_TYPE,
) -> None:
    center = int(center_x), int(center_y)
    axes = int(axes_x), int(axes_y)
    cv2.ellipse(
        image,
        center,
        axes,
        angle,
        start_angle,
        end_angle,
        color,
        thickness,
        line_type,
    )


def draw_image(
    canvas: Image,
    src: Image,
    x: Number,
    y: Number,
) -> None:
    canvas_height = canvas.shape[0]
    canvas_width = canvas.shape[1]
    src_height = src.shape[0]
    src_width = src.shape[1]
    x1 = max(int(x), 0)
    y1 = max(int(y), 0)
    x2 = min(x1 + src_width, canvas_width)
    y2 = min(y1 + src_height, canvas_height)
    canvas[y1:y2, x1:x2] = src


def draw_crosshair_point(
    image: Image,
    x: Number,
    y: Number,
    radius=CROSSHAIR_POINT_RADIUS,
    thickness=CROSSHAIR_POINT_THICKNESS,
    color=CROSSHAIR_POINT_COLOR,
    line_type=CROSSHAIR_POINT_LINE_TYPE,
    padding=CROSSHAIR_POINT_PADDING,
    circle=True,
) -> None:
    if padding == 0:
        left = x - radius, y
        top = x, y - radius
        right = x + radius, y
        bottom = x, y + radius

        draw_line(image, left, right, color, thickness, line_type)
        draw_line(image, top, bottom, color, thickness, line_type)
    else:
        left1 = x - radius - padding, y
        left2 = x - padding, y

        top1 = x, y - radius - padding
        top2 = x, y - padding

        right1 = x + radius + padding, y
        right2 = x + padding, y

        bottom1 = x, y + radius + padding
        bottom2 = x, y + padding

        draw_line(image, left1, left2, color, thickness, line_type)
        draw_line(image, top1, top2, color, thickness, line_type)
        draw_line(image, right1, right2, color, thickness, line_type)
        draw_line(image, bottom1, bottom2, color, thickness, line_type)

    if circle:
        draw_circle(
            image=image,
            x=x,
            y=y,
            radius=radius,
            color=color,
            thickness=thickness,
            line_type=line_type,
        )


def draw_marker(
    image: Image,
    x: Number,
    y: Number,
    marker_size=MARKER_SIZE,
    marker_type=MarkerType.CROSS,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
) -> None:
    position = int(x), int(y)
    cv2.drawMarker(
        image,
        position,
        color,
        marker_type.value,
        marker_size,
        thickness,
        line_type,
    )


def draw_arrowed(
    image: Image,
    point1: PointT,
    point2: PointT,
    tip_length=ARROWED_LINE_TIP_LENGTH,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
) -> None:
    x1, y1 = int(point1[0]), int(point1[1])
    x2, y2 = int(point2[0]), int(point2[1])
    cv2.arrowedLine(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
        line_type,
        tipLength=tip_length,
    )


def draw_text(
    image: Image,
    text: str,
    x: Number,
    y: Number,
    font=FONT,
    scale=FONT_SCALE,
    color=COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
    origin=DrawTextOrigin.TOP_LEFT,
) -> None:
    org = int(x), int(y)
    cv2.putText(
        image,
        text,
        org,
        font,
        scale,
        color,
        thickness,
        line_type,
        origin.value,
    )


def draw_outline_text(
    image: Image,
    text: str,
    x: Number,
    y: Number,
    font=FONT,
    scale=FONT_SCALE,
    fill_color=OUTLINE_FILL_COLOR,
    outline_color=OUTLINE_COLOR,
    thickness=THICKNESS,
    outline_thickness=OUTLINE_THICKNESS,
    line_type=LINE_TYPE,
    origin=DrawTextOrigin.TOP_LEFT,
) -> None:
    bg_color = outline_color
    fg_color = fill_color
    bg_thickness = outline_thickness
    fg_thickness = thickness
    draw_text(image, text, x, y, font, scale, bg_color, bg_thickness, line_type, origin)
    draw_text(image, text, x, y, font, scale, fg_color, fg_thickness, line_type, origin)


def get_font_scale_from_height(
    pixel_height: int,
    font=FONT,
    thickness=THICKNESS,
) -> float:
    return cv2.getFontScaleFromHeight(font, pixel_height, thickness)


def get_text_size(
    text: str,
    font=FONT,
    scale=FONT_SCALE,
    thickness=THICKNESS,
) -> Tuple[SizeInt, int]:
    text_size = cv2.getTextSize(text, font, scale, thickness)
    text_width, text_height = text_size[0]
    baseline = text_size[1]
    return (text_width, text_height), baseline


def measure_multiline_text_box_size(
    text: str,
    font=FONT,
    scale=FONT_SCALE,
    thickness=THICKNESS,
    linefeed=MULTILINE_LINEFEED,
    line_spacing=MULTILINE_LINE_SPACING,
) -> Tuple[int, int, List[Tuple[str, SizeInt, int]]]:
    tws = list()
    ths = list()
    lines = list()
    for line in text.split(linefeed):
        text_size = cv2.getTextSize(line, font, scale, thickness)
        text_width, text_height = text_size[0]
        baseline = text_size[1]
        line_height = text_height + baseline + line_spacing
        tws.append(text_width)
        ths.append(line_height)
        lines.append((line, (text_width, text_height), baseline))
    box_width = max(tws)
    box_height = sum(ths)
    box_height -= line_spacing  # The last line has no bottom line spacing.
    return box_width, box_height, lines


def draw_multiline_text_with_lines(
    image: Image,
    lines: List[Tuple[str, SizeInt, int]],
    x: Number,
    y: Number,
    font=FONT,
    scale=FONT_SCALE,
    color=MULTILINE_COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
    line_spacing=MULTILINE_LINE_SPACING,
) -> None:
    for line in lines:
        text = line[0]
        width, height = line[1]
        baseline = line[2]
        y += height
        draw_text(
            image,
            text,
            x,
            y,
            font,
            scale,
            color,
            thickness,
            line_type,
            DrawTextOrigin.TOP_LEFT,
        )
        y += baseline + line_spacing


def draw_multiline_text(
    image: Image,
    text: str,
    x: Number,
    y: Number,
    font=FONT,
    scale=FONT_SCALE,
    color=MULTILINE_COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
    linefeed=MULTILINE_LINEFEED,
    line_spacing=MULTILINE_LINE_SPACING,
) -> None:
    width, height, lines = measure_multiline_text_box_size(
        text,
        font,
        scale,
        thickness,
        linefeed,
        line_spacing,
    )
    draw_multiline_text_with_lines(
        image,
        lines,
        x,
        y,
        font,
        scale,
        color,
        thickness,
        line_type,
        line_spacing,
    )


def draw_multiline_text_box(
    image: Image,
    text: str,
    x: Number,
    y: Number,
    font=FONT,
    scale=FONT_SCALE,
    color=MULTILINE_COLOR,
    thickness=THICKNESS,
    line_type=LINE_TYPE,
    linefeed=MULTILINE_LINEFEED,
    line_spacing=MULTILINE_LINE_SPACING,
    background_color=MULTILINE_BACKGROUND_COLOR,
    background_alpha=MULTILINE_BACKGROUND_ALPHA,
    margin=MULTILINE_BOX_MARGIN,
    anchor_x=MULTILINE_BOX_ANCHOR_X,
    anchor_y=MULTILINE_BOX_ANCHOR_Y,
) -> RectInt:
    assert 0 <= anchor_x <= 1
    assert 0 <= anchor_y <= 1
    assert 0 <= background_alpha <= 1

    ch = image.shape[0]
    cw = image.shape[1]

    bw, bh, lines = measure_multiline_text_box_size(
        text, font, scale, thickness, linefeed, line_spacing
    )
    bw += margin * 2
    bh += margin * 2
    box = full((bh, bw, 3), background_color, dtype=uint8)

    bx = bw * anchor_x
    by = bh * anchor_y
    x1 = max(int((x + cw * anchor_x) - bx), 0)
    y1 = max(int((y + ch * anchor_y) - by), 0)
    x2 = min(x1 + bw, cw)
    y2 = min(y1 + bh, ch)
    w = x2 - x1
    h = y2 - y1

    assert 0 <= x1 <= cw
    assert 0 <= y1 <= ch
    assert 0 <= x2 <= cw
    assert 0 <= y2 <= ch
    assert w <= bw
    assert h <= bh

    img_area = image[y1:y2, x1:x2]
    box_area = box[0:h, 0:w]

    alpha = background_alpha
    beta = 1.0 - background_alpha
    if alpha >= 1:
        mixed = box_area
    elif beta >= 1:
        mixed = img_area
    else:
        mixed = cv2.addWeighted(box_area, alpha, img_area, beta, 0)

    draw_multiline_text_with_lines(
        mixed,
        lines,
        x + margin,
        y + margin,
        font,
        scale,
        color,
        thickness,
        line_type,
        line_spacing,
    )
    image[y1:y2, x1:x2] = mixed
    return x1, y1, x2, y2


class CvlDrawable:
    @staticmethod
    def cvl_draw_point(
        image: Image,
        x: Number,
        y: Number,
        radius=RADIUS,
        color=COLOR,
        thickness=FILLED,
        line_type=LINE_AA,
    ):
        return draw_point(image, x, y, radius, color, thickness, line_type)

    @staticmethod
    def cvl_draw_line(
        image: Image,
        point1: PointT,
        point2: PointT,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
    ):
        return draw_line(image, point1, point2, color, thickness, line_type)

    @staticmethod
    def cvl_draw_rectangle(
        image: Image,
        roi: RectT,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
    ):
        return draw_rectangle(image, roi, color, thickness, line_type)

    @staticmethod
    def cvl_draw_circle(
        image: Image,
        x: Number,
        y: Number,
        radius=RADIUS,
        color=COLOR,
        thickness=FILLED,
        line_type=LINE_TYPE,
    ):
        return draw_circle(image, x, y, radius, color, thickness, line_type)

    @staticmethod
    def cvl_draw_image(
        canvas: Image,
        src: Image,
        x: Number,
        y: Number,
    ):
        return draw_image(canvas, src, x, y)

    @staticmethod
    def cvl_draw_crosshair_point(
        image: Image,
        x: Number,
        y: Number,
        radius=CROSSHAIR_POINT_RADIUS,
        thickness=CROSSHAIR_POINT_THICKNESS,
        color=CROSSHAIR_POINT_COLOR,
        line_type=CROSSHAIR_POINT_LINE_TYPE,
        padding=CROSSHAIR_POINT_PADDING,
        circle=True,
    ):
        return draw_crosshair_point(
            image, x, y, radius, thickness, color, line_type, padding, circle
        )

    @staticmethod
    def cvl_draw_marker(
        image: Image,
        x: Number,
        y: Number,
        marker_type=MarkerType.CROSS,
        marker_size=MARKER_SIZE,
        color=CROSSHAIR_POINT_COLOR,
        thickness=CROSSHAIR_POINT_THICKNESS,
        line_type=CROSSHAIR_POINT_LINE_TYPE,
    ):
        return draw_marker(
            image,
            x,
            y,
            marker_size,
            marker_type,
            color,
            thickness,
            line_type,
        )

    @staticmethod
    def cvl_draw_arrowed(
        image: Image,
        point1: PointT,
        point2: PointT,
        tip_length=ARROWED_LINE_TIP_LENGTH,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
    ):
        return draw_arrowed(
            image,
            point1,
            point2,
            tip_length,
            color,
            thickness,
            line_type,
        )

    @staticmethod
    def cvl_draw_text(
        image: Image,
        text: str,
        x: Number,
        y: Number,
        font=FONT,
        scale=FONT_SCALE,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
        origin=DrawTextOrigin.TOP_LEFT,
    ):
        return draw_text(
            image,
            text,
            x,
            y,
            font,
            scale,
            color,
            thickness,
            line_type,
            origin,
        )

    @staticmethod
    def cvl_draw_outline_text(
        image: Image,
        text: str,
        x: Number,
        y: Number,
        font=FONT,
        scale=FONT_SCALE,
        fill_color=OUTLINE_FILL_COLOR,
        outline_color=OUTLINE_COLOR,
        thickness=THICKNESS,
        outline_thickness=OUTLINE_THICKNESS,
        line_type=LINE_TYPE,
    ):
        return draw_outline_text(
            image,
            text,
            x,
            y,
            font,
            scale,
            fill_color,
            outline_color,
            thickness,
            outline_thickness,
            line_type,
        )

    @staticmethod
    def cvl_get_font_scale_from_height(
        pixel_height: int,
        font=FONT,
        thickness=THICKNESS,
    ):
        return get_font_scale_from_height(font, pixel_height, thickness)

    @staticmethod
    def cvl_get_text_size(
        text: str,
        font=FONT,
        scale=FONT_SCALE,
        thickness=THICKNESS,
    ):
        return get_text_size(text, font, scale, thickness)

    @staticmethod
    def cvl_measure_multiline_text_box_size(
        text: str,
        font=FONT,
        scale=FONT_SCALE,
        thickness=THICKNESS,
        linefeed=MULTILINE_LINEFEED,
        line_spacing=MULTILINE_LINE_SPACING,
    ):
        return measure_multiline_text_box_size(
            text,
            font,
            scale,
            thickness,
            linefeed,
            line_spacing,
        )

    @staticmethod
    def cvl_draw_multiline_text_with_lines(
        image: Image,
        lines: List[Tuple[str, SizeInt, int]],
        x: Number,
        y: Number,
        font=FONT,
        scale=FONT_SCALE,
        color=MULTILINE_COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
        line_spacing=MULTILINE_LINE_SPACING,
    ):
        return draw_multiline_text_with_lines(
            image,
            lines,
            x,
            y,
            font,
            scale,
            color,
            thickness,
            line_type,
            line_spacing,
        )

    @staticmethod
    def cvl_draw_multiline_text(
        image: Image,
        text: str,
        x: Number,
        y: Number,
        font=FONT,
        scale=FONT_SCALE,
        color=MULTILINE_COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
        linefeed=MULTILINE_LINEFEED,
    ):
        return draw_multiline_text(
            image,
            text,
            x,
            y,
            font,
            scale,
            color,
            thickness,
            line_type,
            linefeed,
        )

    @staticmethod
    def cvl_draw_multiline_text_box(
        image: Image,
        text: str,
        x: Number,
        y: Number,
        font=FONT,
        scale=FONT_SCALE,
        color=MULTILINE_COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
        linefeed=MULTILINE_LINEFEED,
        line_spacing=MULTILINE_LINE_SPACING,
        background_color=MULTILINE_BACKGROUND_COLOR,
        background_alpha=MULTILINE_BACKGROUND_ALPHA,
        margin=MULTILINE_BOX_MARGIN,
        anchor_x=MULTILINE_BOX_ANCHOR_X,
        anchor_y=MULTILINE_BOX_ANCHOR_Y,
    ):
        return draw_multiline_text_box(
            image,
            text,
            x,
            y,
            font,
            scale,
            color,
            thickness,
            line_type,
            linefeed,
            line_spacing,
            background_color,
            background_alpha,
            margin,
            anchor_x,
            anchor_y,
        )

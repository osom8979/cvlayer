# -*- coding: utf-8 -*-

from typing import Final, Optional, Tuple

import cv2

from cvlayer.typing import Color, Image, Number, PointT, RectT

FILLED: Final[int] = cv2.FILLED

LINE_4: Final[int] = cv2.LINE_4  # Bresenham 4 Connect
LINE_8: Final[int] = cv2.LINE_8  # Bresenham 8 Connect
LINE_AA: Final[int] = cv2.LINE_AA  # Anti-Aliasing

MARKER_CROSS: Final[int] = cv2.MARKER_CROSS
MARKER_TILTED_CROSS: Final[int] = cv2.MARKER_TILTED_CROSS
MARKER_STAR: Final[int] = cv2.MARKER_STAR
MARKER_DIAMOND: Final[int] = cv2.MARKER_DIAMOND
MARKER_SQUARE: Final[int] = cv2.MARKER_SQUARE
MARKER_TRIANGLE_UP: Final[int] = cv2.MARKER_TRIANGLE_UP
MARKER_TRIANGLE_DOWN: Final[int] = cv2.MARKER_TRIANGLE_DOWN

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
MULTILINE_BACKGROUND_COLOR: Final[Color] = (0, 0, 0)
MULTILINE_BACKGROUND_ALPHA: Final[float] = 0.4
MULTILINE_MARGIN: Final[int] = 8

CROSSHAIR_POINT_RADIUS: Final[int] = 6
CROSSHAIR_POINT_THICKNESS: Final[int] = 1
CROSSHAIR_POINT_COLOR: Final[Color] = (0, 0, 255)
CROSSHAIR_POINT_LINE_TYPE: Final[int] = LINE_AA
CROSSHAIR_POINT_PADDING: Final[int] = 2


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


def draw_crosshair_point(
    image: Image,
    point: PointT,
    radius=CROSSHAIR_POINT_RADIUS,
    thickness=CROSSHAIR_POINT_THICKNESS,
    color=CROSSHAIR_POINT_COLOR,
    line_type=CROSSHAIR_POINT_LINE_TYPE,
    padding=CROSSHAIR_POINT_PADDING,
    circle=True,
) -> None:
    x, y = point

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
) -> None:
    org = int(x), int(y)
    bg_color = outline_color
    fg_color = fill_color
    bg_thickness = outline_thickness
    fg_thickness = thickness
    cv2.putText(image, text, org, font, scale, bg_color, bg_thickness, line_type)
    cv2.putText(image, text, org, font, scale, fg_color, fg_thickness, line_type)


def measure_multiline_text_box_size(
    text: str,
    font=FONT,
    scale=FONT_SCALE,
    thickness=THICKNESS,
    linefeed=MULTILINE_LINEFEED,
) -> Tuple[int, int]:
    tws = []
    ths = []
    for line in text.split(linefeed):
        text_size = cv2.getTextSize(line, font, scale, thickness)
        tw, th = text_size[0]
        baseline = text_size[1]
        tws.append(tw)
        ths.append(th + baseline + (thickness * 2))
    return max(tws), sum(ths)


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
    background_color=MULTILINE_BACKGROUND_COLOR,
    background_alpha=MULTILINE_BACKGROUND_ALPHA,
    margin=MULTILINE_MARGIN,
    canvas_width: Optional[int] = None,
    canvas_height: Optional[int] = None,
) -> None:
    box_width, box_height = measure_multiline_text_box_size(
        text, font, scale, thickness, linefeed
    )

    left_offset = canvas_width - box_width + x if canvas_width and x < 0 else x
    top_offset = canvas_height - box_height + y if canvas_height and y < 0 else y

    overlay = image.copy()
    x1 = left_offset
    y1 = top_offset
    x2 = left_offset + box_width + (margin * 2)
    y2 = top_offset + box_height + (margin * 2)
    p1 = (x1, y1)
    p2 = (x2, y2)
    cv2.rectangle(overlay, p1, p2, background_color, FILLED)

    alpha = background_alpha
    beta = 1.0 - background_alpha
    gamma = 0
    image[::] = cv2.addWeighted(overlay, alpha, image, beta, gamma)

    next_y = top_offset + margin
    for line in text.split(linefeed):
        text_size = cv2.getTextSize(line, font, scale, thickness)
        tw, th = text_size[0]
        baseline = text_size[1]  # noqa
        pos_x = int(left_offset + margin)
        pos_y = int(next_y + th + thickness + baseline)
        pos = pos_x, pos_y
        cv2.putText(image, line, pos, font, scale, color, thickness, line_type)
        next_y += th + baseline + (thickness * 2)


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
        draw_point(image, x, y, radius, color, thickness, line_type)

    @staticmethod
    def cvl_draw_line(
        image: Image,
        point1: PointT,
        point2: PointT,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
    ):
        draw_line(image, point1, point2, color, thickness, line_type)

    @staticmethod
    def cvl_draw_rectangle(
        image: Image,
        roi: RectT,
        color=COLOR,
        thickness=THICKNESS,
        line_type=LINE_TYPE,
    ):
        draw_rectangle(image, roi, color, thickness, line_type)

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
        draw_circle(image, x, y, radius, color, thickness, line_type)

    @staticmethod
    def cvl_draw_crosshair_point(
        image: Image,
        point: PointT,
        radius=CROSSHAIR_POINT_RADIUS,
        thickness=CROSSHAIR_POINT_THICKNESS,
        color=CROSSHAIR_POINT_COLOR,
        line_type=CROSSHAIR_POINT_LINE_TYPE,
        padding=CROSSHAIR_POINT_PADDING,
        circle=True,
    ) -> None:
        draw_crosshair_point(
            image, point, radius, thickness, color, line_type, padding, circle
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
        draw_outline_text(
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
    def cvl_measure_multiline_text_box_size(
        text: str,
        font=FONT,
        scale=FONT_SCALE,
        thickness=THICKNESS,
        linefeed=MULTILINE_LINEFEED,
    ):
        return measure_multiline_text_box_size(text, font, scale, thickness, linefeed)

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
        background_color=MULTILINE_BACKGROUND_COLOR,
        background_alpha=MULTILINE_BACKGROUND_ALPHA,
        margin=MULTILINE_MARGIN,
        canvas_width: Optional[int] = None,
        canvas_height: Optional[int] = None,
    ):
        draw_multiline_text(
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
            background_color,
            background_alpha,
            margin,
            canvas_width,
            canvas_height,
        )

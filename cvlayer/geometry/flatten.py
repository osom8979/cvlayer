# -*- coding: utf-8 -*-

from cvlayer.typing import LineT


def flatten_line(line: LineT):
    p1, p2 = line
    x1, y1 = p1
    x2, y2 = p2
    return x1, y1, x2, y2

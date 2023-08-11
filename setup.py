# -*- coding: utf-8 -*-

import os
from re import Pattern
from re import compile as re_compile
from typing import Final, List

from setuptools import setup

COMMENT_PATTERN: Final[Pattern] = re_compile(r"#.*$")
SOURCE_PATH = os.path.abspath(__file__)
SOURCE_DIR = os.path.dirname(SOURCE_PATH)
REQUIREMENTS_TEST = os.path.join(SOURCE_DIR, "requirements.test.txt")


def install_requires(file: str, encoding="utf-8") -> List[str]:
    with open(file, encoding=encoding) as f:
        content = f.read()
    lines0 = content.split("\n")
    lines1 = map(lambda x: COMMENT_PATTERN.sub("", x).strip(), lines0)
    lines2 = filter(lambda x: x, lines1)
    return list(lines2)


if __name__ == "__main__":
    setup(tests_require=install_requires(REQUIREMENTS_TEST))

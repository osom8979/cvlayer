# -*- coding: utf-8 -*-

from typing import Dict, List

from cvlayer.keymap.create import (
    DEFAULT_CALLBACK_NAME_PREFIX,
    DEFAULT_CALLBACK_NAME_SUFFIX,
    create_callable_keymap,
)


class CvlKeymap:
    @staticmethod
    def cvl_create_callable_keymap(
        obj: object,
        keymaps: Dict[str, List[int]],
        callback_name_prefix=DEFAULT_CALLBACK_NAME_PREFIX,
        callback_name_suffix=DEFAULT_CALLBACK_NAME_SUFFIX,
    ):
        return create_callable_keymap(
            obj,
            keymaps,
            callback_name_prefix,
            callback_name_suffix,
        )

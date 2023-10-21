# -*- coding: utf-8 -*-

from functools import wraps


def _fake_override(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


try:
    from overrides import override
except ImportError:
    override = _fake_override  # type: ignore[assignment]

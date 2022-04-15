#!/usr/bin/env python3

from typing import Any, Union, Tuple, Type


def assert_type(value: Any, allowed: Union[Type, Tuple[Type, ...]]) -> None:
    """
    Small helper to type check.
    """
    if isinstance(value, allowed):
        return
    if isinstance(allowed, Type):
        str_allowed = allowed.__name__
    else:
        str_allowed = ", ".join([_type.__name__ for _type in allowed])
    raise TypeError(f"Unexpected type: {type(value)}. Expected {str_allowed}.")

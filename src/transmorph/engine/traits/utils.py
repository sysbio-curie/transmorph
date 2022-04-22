#!/usr/bin/env python3

from typing import Any, Tuple, Type, Union


def assert_trait(obj: Any, traits: Union[Type, Tuple[Type, ...]]):
    """
    Raises an exception if $obj is not endowed with the
    trait $trait.
    """
    if isinstance(obj, traits):
        return
    if isinstance(traits, Type):
        all_traits: str = traits.__name__
    else:
        all_traits: str = ", ".join([trait.__name__ for trait in traits])
    raise TypeError(
        f"Object {obj} of type {type(obj)} is not endowed"
        f" with trait(s) {all_traits}."
    )

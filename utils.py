from decimal import Decimal
from typing import Generator, TypeVar


def drange(x: Decimal, y: Decimal, jump: Decimal) -> Generator[float, None, None]:
    while x < y:
        yield float(x)
        x += jump


T = TypeVar("T")


def none_throws(obj: T | None, msg: str | None = None) -> T:
    """
    Raises exception if obj is None, else returns obj.
    """
    if obj is None:
        raise ValueError("Expected non-None value " + (msg or f"for a {T}"))
    return obj

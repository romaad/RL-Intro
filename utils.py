from decimal import Decimal
from typing import Generator


def drange(x: Decimal, y: Decimal, jump: Decimal) -> Generator[float, None, None]:
    while x < y:
        yield float(x)
        x += jump

from typing import Iterable, TypeVar

T = TypeVar("T")


def masked_select(values: list[T], mask: Iterable[bool]) -> list[T]:
    return [v for v, m in zip(values, mask) if m]

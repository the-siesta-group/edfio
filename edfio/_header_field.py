from __future__ import annotations

import datetime
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Generic, TypeVar, overload

T = TypeVar("T", str, int, float, datetime.date, datetime.time)


def encode_str(value: str, length: int) -> bytes:
    if len(value) > length:
        raise ValueError(
            f"{value!r} exceeds maximum field length: {len(value)} > {length}"
        )
    if not value.isprintable():
        raise ValueError(f"{value} contains non-printable characters")
    return value.encode("ascii").ljust(length)


def decode_str(field: bytes) -> str:
    return field.decode().rstrip()


def encode_int(value: int, length: int) -> bytes:
    return encode_str(str(value), length)


def encode_float(value: float, length: int) -> bytes:
    if float(value).is_integer():
        value = int(value)
    return encode_str(str(value), length)


def decode_float(field: bytes) -> float:
    value = float(decode_str(field))
    if math.isinf(value):
        raise ValueError(f"Field value is outside float range: {decode_str(field)}")
    return value


class RawHeaderField(ABC, Generic[T]):
    def __set_name__(self, owner: Any, name: str) -> None:
        self.name = name
        self.private_name = "_" + name

    def __init__(self, length: int, *, is_settable: bool) -> None:
        self.length = length
        self.is_settable = is_settable

    @overload
    def __get__(self, instance: None, owner: Any) -> RawHeaderField[T]:
        ...

    @overload
    def __get__(self, instance: Any, owner: Any) -> T:
        ...

    def __get__(self, instance: Any, owner: Any = None) -> RawHeaderField[T] | T:
        if instance is None:
            return self
        return self.decode(getattr(instance, self.private_name))

    def __set__(self, instance: Any, value: T) -> None:
        if not self.is_settable:
            raise AttributeError(f"can't set attribute {self.name}")
        setattr(instance, self.private_name, self.encode(value))

    @abstractmethod
    def decode(self, field: bytes) -> T:
        raise NotImplementedError

    @abstractmethod
    def encode(self, value: T) -> bytes:
        raise NotImplementedError


class RawHeaderFieldStr(RawHeaderField[str]):
    def __init__(self, length: int, *, is_settable: bool = False) -> None:
        super().__init__(length, is_settable=is_settable)

    def decode(self, field: bytes) -> str:
        return decode_str(field)

    def encode(self, value: str) -> bytes:
        return encode_str(value, self.length)


class RawHeaderFieldInt(RawHeaderField[int]):
    def __init__(self, length: int, *, is_settable: bool = False) -> None:
        super().__init__(length, is_settable=is_settable)

    def decode(self, field: bytes) -> int:
        return int(decode_str(field))

    def encode(self, value: int) -> bytes:
        return encode_int(value, self.length)


class RawHeaderFieldFloat(RawHeaderField[float]):
    def __init__(self, length: int, *, is_settable: bool = False) -> None:
        super().__init__(length, is_settable=is_settable)

    def decode(self, field: bytes) -> float:
        return decode_float(field)

    def encode(self, value: float) -> bytes:
        return encode_float(value, self.length)


class RawHeaderFieldDate(RawHeaderField[datetime.date]):
    def __init__(self, length: int, *, is_settable: bool = False) -> None:
        super().__init__(length, is_settable=is_settable)

    def decode(self, field: bytes) -> datetime.date:
        date = decode_str(field)
        one_or_two_digits = "(\\d{1,2})"
        separator = "[.:'\\-\\/]"
        pattern = re.compile(
            f"""
            {one_or_two_digits}  # day
            {separator}
            {one_or_two_digits}  # month
            {separator}
            {one_or_two_digits}  # year
            """,
            re.VERBOSE,
        )
        date = date.replace(" ", "")
        match = re.fullmatch(pattern, date)
        if match is None:
            raise ValueError(f"Invalid date for format DD.MM.YY: {date!r}")
        day, month, year = (int(g) for g in match.groups())
        if year >= 85:  # noqa: PLR2004
            year += 1900
        else:
            year += 2000
        return datetime.date(year, month, day)

    def encode(self, value: datetime.date) -> bytes:
        if not 1985 <= value.year <= 2084:  # noqa: PLR2004
            raise ValueError("EDF only allows dates from 1985 to 2084")
        return encode_str(value.strftime("%d.%m.%y"), self.length)


class RawHeaderFieldTime(RawHeaderField[datetime.time]):
    def __init__(self, length: int, *, is_settable: bool = False) -> None:
        super().__init__(length, is_settable=is_settable)

    def decode(self, field: bytes) -> datetime.time:
        hours, minutes, seconds = (int(n) for n in decode_str(field).split("."))
        return datetime.time(hours, minutes, seconds)

    def encode(self, value: datetime.time) -> bytes:
        return encode_str(value.isoformat().replace(":", "."), self.length)


def get_header_fields(cls: type) -> Iterator[tuple[str, int]]:
    for name, value in cls.__dict__.items():
        if isinstance(value, RawHeaderField):
            yield name, value.length

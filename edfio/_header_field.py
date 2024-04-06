from __future__ import annotations

import datetime
import math
import re

_one_or_two_digits = "([ ]?\\d{1,2})"
_separator = "[.:'\\-\\/ ]"
DATE_OR_TIME_PATTERN = re.compile(
    f"""
        {_one_or_two_digits}  # day/hour
        {_separator}
        {_one_or_two_digits}  # month/minute
        {_separator}
        {_one_or_two_digits}  # year/second
        """,
    re.VERBOSE,
)


def encode_str(value: str, length: int) -> bytes:
    if len(value) > length:
        raise ValueError(
            f"{value!r} exceeds maximum field length: {len(value)} > {length}"
        )
    if not value.isprintable():
        raise ValueError(f"{value} contains non-printable characters")
    return value.encode("ascii").ljust(length)


def decode_str(field: bytes) -> str:
    return field.decode(errors="replace").rstrip()


def encode_int(value: int, length: int) -> bytes:
    return encode_str(str(value), length)


def encode_float(value: float) -> bytes:
    if float(value).is_integer():
        value = int(value)
    return encode_str(str(value), 8)


def decode_float(field: bytes) -> float:
    value = float(decode_str(field))
    if math.isinf(value):
        raise ValueError(f"Field value is outside float range: {decode_str(field)}")
    return value


def decode_date(field: bytes) -> datetime.date:
    date = decode_str(field)
    match = DATE_OR_TIME_PATTERN.fullmatch(date)
    if match is None:
        raise ValueError(f"Invalid date for format DD.MM.YY: {date!r}")
    day, month, year = (int(g) for g in match.groups())
    if year >= 85:  # noqa: PLR2004
        year += 1900
    else:
        year += 2000
    return datetime.date(year, month, day)


def encode_date(value: datetime.date) -> bytes:
    if not 1985 <= value.year <= 2084:  # noqa: PLR2004
        raise ValueError("EDF only allows dates from 1985 to 2084")
    return encode_str(value.strftime("%d.%m.%y"), 8)


def decode_time(field: bytes) -> datetime.time:
    time = decode_str(field)
    match = DATE_OR_TIME_PATTERN.fullmatch(time)
    if match is None:
        raise ValueError(f"Invalid time for format hh.mm.ss: {time!r}")
    hours, minutes, seconds = (int(g) for g in match.groups())
    return datetime.time(hours, minutes, seconds)


def encode_time(value: datetime.time) -> bytes:
    return encode_str(value.isoformat().replace(":", "."), 8)

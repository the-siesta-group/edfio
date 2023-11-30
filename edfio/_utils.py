from __future__ import annotations

import datetime
import inspect
from typing import Any, Callable, NamedTuple


def calculate_gain_and_offset(
    digital_min: int,
    digital_max: int,
    physical_min: float,
    physical_max: float,
) -> tuple[float, float]:
    gain = (physical_max - physical_min) / (digital_max - digital_min)
    offset = physical_max / gain - digital_max
    return gain, offset


def repr_from_init(obj: Any) -> str:
    parameters = []
    for name in inspect.signature(obj.__class__).parameters:
        parameters.append(f"{name}={getattr(obj, name)!r}")
    return f"{obj.__class__.__name__}({', '.join(parameters)})"


class IntRange(NamedTuple):
    min: int
    max: int


class FloatRange(NamedTuple):
    min: float
    max: float


_MONTH_NAMES = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)


def decode_edfplus_date(date: str) -> datetime.date:
    day, month, year = date.split("-")
    try:
        month_int = _MONTH_NAMES.index(month) + 1
    except ValueError:
        raise ValueError(f"Invalid month: {month}, options: {_MONTH_NAMES}") from None
    return datetime.date(int(year), month_int, int(day))


def encode_edfplus_date(date: datetime.date) -> str:
    return f"{date.day:02}-{_MONTH_NAMES[date.month - 1]}-{date.year:02}"


def encode_annotation_onset(onset: float) -> str:
    string = f"{onset:+.12f}".rstrip("0")
    if string[-1] == ".":
        return string[:-1]
    return string


def encode_annotation_duration(duration: float) -> str:
    if duration < 0:
        raise ValueError(f"Annotation duration must be positive, is {duration}")
    string = f"{duration:.12f}".rstrip("0")
    if string[-1] == ".":
        return string[:-1]
    return string


def round_float_to_8_characters(
    value: float,
    round_func: Callable[[float], int],
) -> float:
    if isinstance(value, int) or value.is_integer():
        return value
    length = 8
    integer_part_length = str(value).find(".")
    if integer_part_length == length:
        return round_func(value)
    factor = 10 ** (length - 1 - integer_part_length)
    return round_func(value * factor) / factor


def validate_subfields(subfields: dict[str, str]) -> None:
    for key, value in subfields.items():
        if not value:
            raise ValueError(f"Subfield {key} must not be an empty string")
        if " " in value:
            raise ValueError(f"Subfield {key} contains spaces: {value!r}")

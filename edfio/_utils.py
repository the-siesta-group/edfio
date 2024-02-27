from __future__ import annotations

import datetime
import inspect
from typing import Any


def repr_from_init(obj: Any) -> str:
    parameters = []
    for name in inspect.signature(obj.__class__).parameters:
        parameters.append(f"{name}={getattr(obj, name)!r}")
    return f"{obj.__class__.__name__}({', '.join(parameters)})"


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
        month_int = _MONTH_NAMES.index(month.upper()) + 1
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


def validate_subfields(subfields: dict[str, str]) -> None:
    for key, value in subfields.items():
        if not value:
            raise ValueError(f"Subfield {key} must not be an empty string")
        if " " in value:
            raise ValueError(f"Subfield {key} contains spaces: {value!r}")

import datetime

import pytest

from edfio._utils import (
    decode_edfplus_date,
    encode_annotation_duration,
    encode_annotation_onset,
    encode_edfplus_date,
)

VALID_EDFPLUS_DATE_PAIRS = (
    ("02-MAY-1951", datetime.date(1951, 5, 2)),
    ("02-DEC-1951", datetime.date(1951, 12, 2)),
    ("02-AUG-1951", datetime.date(1951, 8, 2)),
    ("02-MAY-2051", datetime.date(2051, 5, 2)),
)


@pytest.mark.parametrize(("string", "datetime_"), VALID_EDFPLUS_DATE_PAIRS)
def test_decode_edfplus_date(string: str, datetime_: datetime.date):
    assert decode_edfplus_date(string) == datetime_


@pytest.mark.parametrize(("string", "datetime_"), VALID_EDFPLUS_DATE_PAIRS)
def test_encode_edfplus_date(string: str, datetime_: datetime.date):
    assert encode_edfplus_date(datetime_) == string


def test_decode_edfplus_date_invalid_month_name():
    with pytest.raises(ValueError, match="Invalid month"):
        decode_edfplus_date("02-MAI-1951")


@pytest.mark.parametrize(
    ("onset", "expected"),
    [
        (0, "+0"),
        (0.0, "+0"),
        (0.1, "+0.1"),
        (0.01, "+0.01"),
        (0.001, "+0.001"),
        (0.0001, "+0.0001"),
        (0.00001, "+0.00001"),
        (0.000001, "+0.000001"),
        (0.0000001, "+0.0000001"),
        (0.00000001, "+0.00000001"),
        (0.00000000001, "+0.00000000001"),
        (100000000000.0, "+100000000000"),
        (-0.1, "-0.1"),
        (-0.0000001, "-0.0000001"),
        (-0.0000000001, "-0.0000000001"),
        (-100000000000.0, "-100000000000"),
    ],
)
def test_encode_annotation_onset(onset: float, expected: str):
    assert encode_annotation_onset(onset) == expected


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (0, "0"),
        (0.0, "0"),
        (0.1, "0.1"),
        (0.01, "0.01"),
        (0.001, "0.001"),
        (0.0001, "0.0001"),
        (0.00001, "0.00001"),
        (0.000001, "0.000001"),
        (0.0000001, "0.0000001"),
        (0.00000000001, "0.00000000001"),
        (100000000000.0, "100000000000"),
    ],
)
def test_encode_annotation_duration(duration: float, expected: str):
    assert encode_annotation_duration(duration) == expected


def test_encode_annotation_duration_raises_error_for_negative_values():
    with pytest.raises(ValueError, match="Annotation duration must be positive, is"):
        encode_annotation_duration(-1)

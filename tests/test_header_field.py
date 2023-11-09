import pytest

from edfio._header_field import (
    RawHeaderFieldDate,
    decode_float,
    encode_float,
    encode_int,
)


def test_encode_int_exceeding_field_length_fails():
    with pytest.raises(ValueError, match="exceeds maximum field length"):
        encode_int(100000000000000000, 8)


def test_encode_float_exceeding_field_length_fails():
    with pytest.raises(ValueError, match="exceeds maximum field length"):
        encode_float(123456789.12345, 8)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.234, b"1.234   "),
        (1.234567, b"1.234567"),
        (12345678, b"12345678"),
    ],
)
def test_encode_float(value: float, expected: bytes):
    assert encode_float(value, 8) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.12345678, b"0.123457"),
        (1.23456789, b"1.234568"),
        (12345.6789, b"12345.68"),
        (1234567.8, b"1234568 "),
        (12345678.9, b"12345679"),
        (-0.987654321, b"-0.98765"),
        (-0.444444, b"-0.44444"),
    ],
)
def test_encode_float_requiring_rounding(value: float, expected: bytes):
    with pytest.warns(UserWarning, match="exceeds maximum field length 8, rounding"):
        assert encode_float(value, 8) == expected


@pytest.mark.parametrize("field", [b"1E2345  ", b"-1E2345 "])
def test_decode_float_exceeding_float_range_fails(field: bytes):
    with pytest.raises(ValueError, match="outside float range"):
        decode_float(field)


@pytest.mark.parametrize(
    "date",
    [
        b"        ",
        b"2.8.2051",
    ],
)
def test_date_decode_invalid_format(date: bytes):
    with pytest.raises(ValueError, match="Invalid date for format"):
        RawHeaderFieldDate(8).decode(date)


def test_date_decode_invalid_day():
    with pytest.raises(ValueError, match="day is out of range for month"):
        RawHeaderFieldDate(8).decode(b"32.08.51")


def test_date_decode_invalid_month():
    with pytest.raises(ValueError, match="month must be in 1..12"):
        RawHeaderFieldDate(8).decode(b"02.13.51")

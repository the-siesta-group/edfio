"""
Tests to verify the adherence to the EDF FAQ: https://www.edfplus.info/specs/edffaq.html
"""

import datetime
from pathlib import Path

import numpy as np
import pytest

from edfio import Edf, EdfSignal, read_edf
from edfio._header_field import decode_date, decode_float, decode_time, encode_date


def test_q1_create_edf_signal_with_non_printable_character_in_label_fails():
    with pytest.raises(ValueError, match="contains non-printable characters"):
        EdfSignal(np.arange(10.1), 1, label="\t")


def test_q1_create_edf_signal_with_utf_8_label_fails():
    with pytest.raises(UnicodeEncodeError, match="'ascii' codec can't encode"):
        EdfSignal(np.arange(10.1), 1, label="SpO₂")


NON_STANDARD_DATES_OR_TIMES = (
    b"02.08.51",
    b"2.8.51  ",
    b"2. 8.51 ",
    b" 2. 8.51",
    b"02:08-51",
    b"02/08'51",
    b"2  8  51",
    b" 2 8  51",
    b" 2  8 51",
    b"2 8 51  ",
)


@pytest.mark.parametrize("field", NON_STANDARD_DATES_OR_TIMES)
def test_q2_date_decode_different_formats(field: bytes):
    assert decode_date(field) == datetime.date(2051, 8, 2)


@pytest.mark.parametrize("field", NON_STANDARD_DATES_OR_TIMES)
def test_q2_time_decode_different_formats(field: bytes):
    assert decode_time(field) == datetime.time(2, 8, 51)


@pytest.mark.parametrize(
    ("field", "date"),
    [
        (b"31.12.85", datetime.date(1985, 12, 31)),
        (b"01.01.84", datetime.date(2084, 1, 1)),
    ],
)
def test_q3_clipping_date(field: bytes, date: datetime.date):
    assert decode_date(field) == date


@pytest.mark.parametrize(
    "date",
    [
        datetime.date(1984, 12, 31),
        datetime.date(2085, 1, 1),
    ],
)
def test_q3_exception_on_date_outside_representable_range(date: datetime.date):
    with pytest.raises(ValueError, match="only allows dates from 1985 to 2084"):
        assert decode_date(encode_date(date)) == date


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (b".5      ", 0.5),
        (b"1E3     ", 1000),
        (b"1e3     ", 1000),
        (b"-1.23E-4", -0.000123),
        (b"-123.456", -123.456),
    ],
)
def test_q7_float_decode_different_formats(field: bytes, value: float):
    assert decode_float(field) == value


def test_q7_float_decode_fails_on_comma():
    with pytest.raises(ValueError, match="could not convert string to float"):
        decode_float(b"-123,456")


def test_q8_read_uncalibrated_signal(tmp_file: Path):
    signal = EdfSignal(np.arange(10), 1)
    signal._physical_min = b"        "
    signal._physical_max = b"        "
    signal._digital_min = b"        "
    signal._digital_max = b"        "
    signal._digital = np.arange(10, dtype=np.int16)
    Edf(signals=[signal]).write(tmp_file)
    edf = read_edf(tmp_file)
    np.testing.assert_equal(edf.signals[0].data, np.arange(10))


def test_q8_edf_signal_where_digital_min_equals_digital_max_data_emits_warning_and_returns_uncalibrated():
    signal = EdfSignal(np.array([-10, 10]), 1, physical_range=(-10, 10))
    signal._digital_min = b"0       "
    signal._digital_max = b"0       "
    with pytest.warns(UserWarning, match="Digital minimum equals .* uncalibrated .*"):
        np.testing.assert_equal(signal.data, np.array([-32768, 32767]))

def test_q8_edf_signal_where_physical_min_equals_physical_max_data_emits_warning_and_returns_uncalibrated():
    signal = EdfSignal(np.array([-10, 10]), 1, physical_range=(-10, 10))
    signal._physical_min = b"0       "
    signal._physical_max = b"0       "
    with pytest.warns(UserWarning, match="Physical minimum equals .* uncalibrated .*"):
        np.testing.assert_equal(signal.data, np.array([-32768, 32767]))


@pytest.mark.parametrize(
    ("sampling_frequency", "num_samples", "expected_data_record_duration"),
    [
        (1 / 30, 5, 30),
        (999.98, 49999, 50),
    ],
)
def test_q9_non_integer_sampling_frequency(
    tmp_file: Path,
    sampling_frequency: float,
    num_samples: int,
    expected_data_record_duration: int,
):
    signal = EdfSignal(np.arange(num_samples), sampling_frequency)
    Edf(signals=[signal]).write(tmp_file)
    edf = read_edf(tmp_file)
    assert edf.data_record_duration == expected_data_record_duration


def test_q11_read_non_standard_ascii_characters_in_header(tmp_file: Path):
    edf = Edf(signals=[EdfSignal(np.arange(10), 1)])
    edf._local_patient_identification = "ÄÖÜ".encode().ljust(80)
    edf.write(tmp_file)
    read_edf(tmp_file)


def test_q11_num_data_records_not_specified(tmp_file: Path):
    edf = Edf(signals=[EdfSignal(np.arange(10), 1)])
    edf._num_data_records = b"-1      "
    with pytest.warns(UserWarning, match="num_data_records=-1, determining correct"):
        edf.write(tmp_file)
    edf = read_edf(tmp_file)
    assert edf.num_data_records == -1


def test_q12_sampling_frequency_below_1hz(tmp_file: Path):
    Edf(signals=[EdfSignal(np.arange(10), 0.1)]).write(tmp_file)
    edf = read_edf(tmp_file)
    assert edf.data_record_duration == 10

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from edfio import EdfSignal
from edfio.edf_signal import _FloatRange, _IntRange, _round_float_to_8_characters


# fmt: off
@pytest.mark.parametrize(
    (   "value",       "expected_round", "expected_floor", "expected_ceil"),
    [
        (1.1111114,     1.111111,         1.111111,         1.111112,),
        (1.111111444,   1.111111,         1.111111,         1.111112,),
        (1.111111499,   1.111111,         1.111111,         1.111112,),
        (1.1111115,     1.111112,         1.111111,         1.111112,),
        (1111111.4,     1111111,          1111111,          1111112,),
        (1111111.444,   1111111,          1111111,          1111112,),
        (1111111.499,   1111111,          1111111,          1111112,),
        (1111111.5,     1111112,          1111111,          1111112,),
        (11111111.4,    11111111,         11111111,         11111112,),
        (11111111.444,  11111111,         11111111,         11111112,),
        (11111111.499,  11111111,         11111111,         11111112,),
        (11111111.5,    11111112,         11111111,         11111112,),
        (-1.111114,     -1.11111,         -1.11112,         -1.11111,),
        (-1.11111444,   -1.11111,         -1.11112,         -1.11111,),
        (-1.11111499,   -1.11111,         -1.11112,         -1.11111,),
        (-1.111115,     -1.11112,         -1.11112,         -1.11111,),
        (-111111.4,     -111111,          -111112,          -111111,),
        (-111111.444,   -111111,          -111112,          -111111,),
        (-111111.499,   -111111,          -111112,          -111111,),
        (-111111.5,     -111112,          -111112,          -111111,),
        (-1111111.4,    -1111111,         -1111112,         -1111111,),
        (-1111111.444,  -1111111,         -1111112,         -1111111,),
        (-1111111.499,  -1111111,         -1111112,         -1111111,),
        (-1111111.5,    -1111112,         -1111112,         -1111111,),
    ],
)
# fmt: on
def test_round_float_to_8_characters(
    value: float,
    expected_round: float,
    expected_floor: float,
    expected_ceil: float,
):
    assert _round_float_to_8_characters(value, round) == expected_round
    assert _round_float_to_8_characters(value, math.floor) == expected_floor
    assert _round_float_to_8_characters(value, math.ceil) == expected_ceil


def sine(duration, f, fs):
    t = np.arange(duration * fs) / fs
    return np.sin(2 * np.pi * f * t)


@pytest.fixture()
def dummy_edf_signal() -> EdfSignal:
    data = sine(5, 2, 128)
    return EdfSignal(data, 128)


def test_edf_signal_init_minimal():
    data = sine(5, 2, 128)
    sig = EdfSignal(data, 128)
    tolerance = np.ptp(data) * 2**-16
    np.testing.assert_allclose(
        sig.data,
        data,
        atol=tolerance,
    )
    assert sig.sampling_frequency == 128
    assert sig.physical_min == data.min()
    assert sig.physical_max == data.max()


def test_edf_signal_init_all_parameters():
    data = sine(5, 2, 128)
    physical_range = _FloatRange(-500, 500)
    digital_range = _IntRange(-2048, 2047)
    params = {
        "sampling_frequency": 128,
        "label": "EEG Fpz-Cz",
        "transducer_type": "AgAgCl electrode",
        "physical_dimension": "uV",
        "physical_range": physical_range,
        "digital_range": digital_range,
        "prefiltering": "HP:0.1Hz LP:75Hz",
    }
    sig = EdfSignal(data, **params)
    tolerance = (physical_range.max - physical_range.min) / (
        digital_range.max - digital_range.min
    )
    np.testing.assert_allclose(
        sig.data,
        data,
        atol=tolerance,
    )
    for name, value in params.items():
        assert getattr(sig, name) == value
    assert sig.reserved == ""


@pytest.mark.parametrize(
    "field_name",
    [
        "samples_per_data_record",
        "reserved",
    ],
)
def test_edf_signal_field_cannot_be_set_publicly(field_name: str):
    signal = EdfSignal(np.arange(10), 1)
    with pytest.raises(AttributeError, match="can't set attribute"):
        setattr(signal, field_name, None)


EDFSIGNAL_SETTER_TEST_FIELDS = [
    ("_label", 16, "EEG FPz-Cz"),
    ("transducer_type", 80, "AgAgCl electrode"),
    ("physical_dimension", 8, "uV"),
    ("prefiltering", 80, "HP:0.1Hz LP:75Hz"),
]


@pytest.mark.parametrize(
    ("field_name", "field_length", "value"),
    EDFSIGNAL_SETTER_TEST_FIELDS,
)
def test_edfsignal_setter_raises_error_if_field_length_is_exceeded(
    dummy_edf_signal: EdfSignal,
    field_name: str,
    field_length: int,
    value: str,
):
    prev_val = getattr(dummy_edf_signal, field_name)
    with pytest.raises(ValueError, match="exceeds maximum field length"):
        setattr(dummy_edf_signal, field_name, "X" * (field_length + 1))
    assert getattr(dummy_edf_signal, field_name) == prev_val


@pytest.mark.parametrize(
    ("field_name", "field_length", "value"),
    EDFSIGNAL_SETTER_TEST_FIELDS,
)
def test_edfsignal_setter_sets_raw_data_fields(
    dummy_edf_signal: EdfSignal,
    field_name: str,
    field_length: int,
    value: str,
):
    setattr(dummy_edf_signal, field_name, value)
    assert getattr(dummy_edf_signal, field_name) == value
    assert getattr(dummy_edf_signal, "_" + field_name) == str(value).ljust(
        field_length
    ).encode("ascii")


def test_edf_signal_samples_per_data_record_not_set(dummy_edf_signal: EdfSignal):
    with pytest.raises(AttributeError, match="no attribute '_samples_per_data_record'"):
        dummy_edf_signal.samples_per_data_record


@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        (EdfSignal(np.arange(5), 1), "<EdfSignal 1Hz>"),
        (EdfSignal(np.arange(5), 1, label="ECG"), "<EdfSignal ECG 1Hz>"),
        (EdfSignal(np.arange(5), 256.0), "<EdfSignal 256Hz>"),
        (EdfSignal(np.arange(5), 123.456), "<EdfSignal 123.456Hz>"),
    ],
)
def test_edf_signal_repr(signal: EdfSignal, expected: str):
    assert repr(signal) == expected


def test_edf_signal_from_raw_header_has_no_data_by_default():
    sig = EdfSignal._from_raw_header(
        20,
        _label=b"".ljust(16),
        transducer_type=b"".ljust(80),
        physical_dimension=b"".ljust(8),
        physical_min=b"-500".ljust(8),
        physical_max=b"500".ljust(8),
        digital_min=b"-2048".ljust(8),
        digital_max=b"2047".ljust(8),
        prefiltering=b"".ljust(80),
        samples_per_data_record=b"1".ljust(8),
        reserved=b"".ljust(32),
    )
    with pytest.raises(AttributeError, match="has no attribute '_digital'"):
        sig.data


@pytest.mark.parametrize(
    ("physical_range", "actual_data"),
    [
        pytest.param(
            _FloatRange(-18.1234, 112.5432),
            [13.0, -17.9, -18.0, 112.543251],
            id="Exceeds maximum",
        ),
        pytest.param(
            _FloatRange(-18.1234, 112.5432),
            [13.0, -18.123451, -18.0, 110.0],
            id="Exceeds minimum",
        ),
    ],
)
def test_write_data_that_exceeds_physical_range(
    physical_range: _FloatRange, actual_data: list[float], tmp_file: Path
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Signal range [{min(actual_data)}, {max(actual_data)}] out of physical range: [{physical_range.min}, {physical_range.max}]"
        ),
    ):
        EdfSignal(
            data=np.asarray(actual_data, dtype="float64"),
            sampling_frequency=1,
            physical_range=physical_range,
        )


def test_edf_signal_with_too_small_physical_range_fails():
    with pytest.raises(ValueError, match="out of physical range"):
        EdfSignal(np.arange(10), 1, physical_range=(3, 5))


@pytest.mark.parametrize("data", [[-3, 3], [-3, 7], [-7, 7]])
def test_edf_signal_update_data(data):
    data = np.array(data)
    signal = EdfSignal(np.array([-5, 5]), 1)
    signal.update_data(data)
    np.testing.assert_array_equal(signal.data, data)
    np.testing.assert_array_equal(signal._digital, np.array([-32768, 32767]))
    assert signal.physical_range == _FloatRange(data.min(), data.max())


@pytest.mark.parametrize("length", [9, 11])
def test_edf_signal_update_data_fails_on_length_mismatch(length: int):
    signal = EdfSignal(np.arange(10), 1)
    with pytest.raises(ValueError, match="Signal lengths must match"):
        signal.update_data(np.arange(length))


def test_edf_signal_update_data_keep_physical_range():
    signal = EdfSignal(
        np.array([90.1, 98.3, 100.0]),
        1,
        physical_range=(0, 100),
        digital_range=(0, 1000),
    )
    new_data = np.array([92.3, 97.5, 99.9])
    signal.update_data(new_data, keep_physical_range=True)
    assert signal.physical_range == (0, 100)
    assert signal.digital_range == (0, 1000)
    np.testing.assert_almost_equal(signal.data, new_data, decimal=14)
    np.testing.assert_array_equal(signal._digital, new_data * 10)


@pytest.mark.parametrize("data", [[-7, 3], [-3, 7], [-7, 7]])
def test_edf_signal_update_data_keep_physical_range_raises_error_if_new_data_exceeds_physical_range(
    data,
):
    signal = EdfSignal(np.array([-5, 5]), 1)
    with pytest.raises(ValueError, match="Signal range .* out of physical range"):
        signal.update_data(np.array(data), keep_physical_range=True)


def test_edf_signal_update_data_resampling():
    signal = EdfSignal(np.arange(10), 10, digital_range=(0, 9), physical_range=(0, 9))
    signal.update_data(
        np.arange(0, 10, 2), sampling_frequency=5, keep_physical_range=True
    )
    assert signal.sampling_frequency == 5
    np.testing.assert_array_equal(signal.data, np.arange(0, 10, 2))


def test_edf_signal_update_data_resampling_noninteger_sampling_rates():
    signal = EdfSignal(
        np.arange(11), 5.5, digital_range=(0, 10), physical_range=(0, 10)
    )
    signal.update_data(np.arange(0, 10), sampling_frequency=5, keep_physical_range=True)
    assert signal.sampling_frequency == 5
    np.testing.assert_array_equal(signal.data, np.arange(0, 10))


def test_edf_signal_update_data_resampling_invalid_duration():
    signal = EdfSignal(np.arange(10), 10, digital_range=(0, 9), physical_range=(0, 9))
    with pytest.raises(ValueError, match="Signal lengths must match:"):
        signal.update_data(np.arange(0, 10, 2.5), sampling_frequency=5)


@pytest.mark.parametrize(
    "sampling_frequency",
    [-10, 0],
)
def test_edf_signal_update_data_resampling_non_positive_sampling_frequency(
    sampling_frequency: float,
):
    signal = EdfSignal(np.arange(10), 10, digital_range=(0, 9), physical_range=(0, 9))
    with pytest.raises(ValueError, match="Sampling frequency must be positive"):
        signal.update_data(np.arange(0, 10, 2), sampling_frequency=sampling_frequency)


def test_edf_signal_update_data_resampling_non_integer_samples():
    signal = EdfSignal(np.arange(10), 10, digital_range=(0, 9), physical_range=(0, 9))
    with pytest.raises(ValueError, match="non-integer number of samples"):
        signal.update_data(np.arange(0, 10, 2), sampling_frequency=9.8)


def test_edf_signal_update_data_resampling_tolerance():
    signal = EdfSignal(np.arange(1000000), 100)
    signal.update_data(np.arange(1000001), sampling_frequency=100.0001)
    assert signal.sampling_frequency == 100.0001
    assert len(signal.data) == 1000001


def test_edf_signal_data_cannot_be_modified(dummy_edf_signal: EdfSignal):
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        dummy_edf_signal.data[5] = -1


@pytest.mark.parametrize("data", [[0, np.nan, 2], [0, np.inf, 2], [0, -np.inf, 2]])
def test_instantiating_edf_signal_with_non_finite_data_raises_error(data: list[float]):
    with pytest.raises(ValueError, match="Signal data must contain only finite values"):
        EdfSignal(np.array(data), 3)


def test_creating_edf_signal_where_physical_min_equals_physical_max_raises_error():
    with pytest.raises(ValueError, match="Physical minimum .* must differ from"):
        EdfSignal(np.zeros(5), sampling_frequency=1, physical_range=(0, 0))


def test_creating_edf_signal_where_digital_min_equals_digital_max_raises_error():
    with pytest.raises(ValueError, match="Digital minimum .* must differ from"):
        EdfSignal(np.arange(5), sampling_frequency=1, digital_range=(0, 0))


def test_creating_edf_signal_with_constant_data_is_possible():
    signal = EdfSignal(np.zeros(5), 1)
    np.testing.assert_array_equal(signal.data, np.zeros(5))


def test_rounding_of_physical_range_does_not_produce_clipping_or_integer_overflow():
    data = np.array([0, 0.0000014999])
    sig = EdfSignal(data, 1)
    np.testing.assert_allclose(sig.data, data, atol=1e-11)
    # round((0.0000014999 - 0.0) / 0.000002 * 65535 + (-32768)) = 16380
    assert sig._digital.tolist() == [-32768, 16380]


def test_edf_signal_init_does_not_accept_edf_annotations_as_label():
    with pytest.raises(ValueError, match="must not be 'EDF Annotations'"):
        EdfSignal(np.arange(2), 1, label="EDF Annotations")


def test_edf_signal_label_cannot_be_set_to_edf_annotations():
    with pytest.raises(ValueError, match="must not be 'EDF Annotations'"):
        EdfSignal(np.arange(2), 1).label = "EDF Annotations"

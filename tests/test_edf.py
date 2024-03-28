from __future__ import annotations

import datetime
import io
import json
import tempfile
from pathlib import Path
from shutil import copyfile
from typing import Literal

import numpy as np
import pytest

from edfio import (
    AnonymizedDateError,
    Edf,
    EdfAnnotation,
    EdfSignal,
    Patient,
    Recording,
    read_edf,
)
from edfio.edf import _calculate_num_data_records
from edfio.edf_annotations import _create_annotations_signal
from edfio.edf_signal import _FloatRange
from tests import TEST_DATA_DIR

EDF_FILE = TEST_DATA_DIR / "short_psg.edf"
EDF_SIGNAL_REFERENCE_FILE = TEST_DATA_DIR / "short_psg_header_reference.json"


def test_read_edf():
    edf = read_edf(EDF_FILE)

    assert edf.version == 0
    assert edf.local_patient_identification == "X F X Female_33yr"
    assert edf.local_recording_identification == "Startdate 24-APR-1989 X X X"
    assert edf.startdate == datetime.date(1989, 4, 24)
    assert edf.starttime == datetime.time(16, 13, 00)
    assert edf.bytes_in_header_record == 2048
    assert edf.reserved == ""
    assert edf.num_data_records == 10
    assert edf.data_record_duration == 30.0
    assert edf.num_signals == 7

    with EDF_SIGNAL_REFERENCE_FILE.open() as jf:
        reference_signal_headers = json.load(jf)
    assert len(edf.signals) == len(reference_signal_headers)
    for signal_header, reference_signal_header in zip(
        edf.signals, reference_signal_headers
    ):
        for key, reference_value in reference_signal_header.items():
            assert getattr(signal_header, key) == reference_value

    signal_checksums = [
        5786.725274725277,
        -10889.671062271369,
        22995.836874236884,
        109827,
        1028.81,
        11160.999999999998,
        265812,
    ]
    assert len(edf.signals) == len(signal_checksums)
    for signal, checksum in zip(edf.signals, signal_checksums):
        np.testing.assert_approx_equal(signal.data.sum(), checksum)


def sine(duration, f, fs):
    t = np.arange(duration * fs) / fs
    return np.sin(2 * np.pi * f * t)


EDF_IDENTIFICATION_SETTER_TEST_FIELDS = [
    ("local_patient_identification", 80, "dummy-patient-id"),
    ("local_recording_identification", 80, "dummy-recording-id"),
]


@pytest.mark.parametrize(
    ("field_name", "field_length", "value"),
    EDF_IDENTIFICATION_SETTER_TEST_FIELDS,
)
def test_edf_identificationsetter_raises_error_if_field_length_is_exceeded(
    field_name: str,
    field_length: int,
    value: str,
):
    edf = read_edf(EDF_FILE)
    prev_val = getattr(edf, field_name)
    with pytest.raises(ValueError, match="exceeds maximum field length"):
        setattr(edf, field_name, "X" * (field_length + 1))
    assert getattr(edf, field_name) == prev_val


@pytest.mark.parametrize(
    ("field_name", "field_length", "value"),
    EDF_IDENTIFICATION_SETTER_TEST_FIELDS,
)
def test_edf_identificationsetter_sets_raw_data_fields(
    field_name: str,
    field_length: int,
    value: str,
):
    edf = read_edf(EDF_FILE)
    setattr(edf, field_name, value)
    assert getattr(edf, field_name) == value
    assert getattr(edf, "_" + field_name) == value.encode("ascii").ljust(field_length)


@pytest.mark.parametrize(
    ("value", "raw"),
    [
        (datetime.date(1985, 1, 15), b"15.01.85"),
        (datetime.date(2084, 1, 15), b"15.01.84"),
    ],
)
def test_edf_startdate_setter(value: datetime.date, raw: bytes):
    edf = read_edf(EDF_FILE)
    edf.startdate = value
    assert edf.startdate == value
    assert edf.__startdate == raw


def test_edf_starttime_setter():
    edf = read_edf(EDF_FILE)

    starttime = datetime.time(20, 10, 42)
    edf.starttime = starttime
    assert edf.starttime == starttime
    assert edf.__starttime == b"20.10.42"


def test_edf_init_minimal():
    Edf([EdfSignal(sine(5, 2, 20), 20, physical_range=(-1, 1))])


def test_edf_init_with_negative_data_record_duration():
    with pytest.raises(ValueError, match="must be positive"):
        Edf(
            [EdfSignal(sine(5, 2, 20), 20, physical_range=(-1, 1))],
            data_record_duration=-1,
        )


@pytest.mark.parametrize(
    ("edf", "expected"),
    [
        (
            Edf([EdfSignal(np.arange(5), 1)], annotations=[EdfAnnotation(0, 1, "A")]),
            "<Edf 1 signal 1 annotation>",
        ),
        (Edf([EdfSignal(np.arange(5), 1)] * 2), "<Edf 2 signals 0 annotations>"),
        (
            Edf([], annotations=[EdfAnnotation(0, 1, "A")] * 2),
            "<Edf 0 signals 2 annotations>",
        ),
    ],
)
def test_edf_repr(edf: Edf, expected: str):
    assert repr(edf) == expected


def test_edf_init_all_parameters():
    local_patient_identification = "X F X Female_33yr"
    local_recording_identification = "Startdate 24-APR-1989 X X X"
    starttime = datetime.time(16, 13, 00)
    data_record_duration = 2.5

    edf = Edf(
        [EdfSignal(sine(5, 2, 20), 20, physical_range=(-1, 1))],
        patient=Patient._from_str(local_patient_identification),
        recording=Recording._from_str(local_recording_identification),
        starttime=starttime,
        data_record_duration=data_record_duration,
    )
    assert len(edf.signals) == 1
    assert edf._version == b"0".ljust(8)
    assert edf._local_patient_identification == local_patient_identification.encode(
        "ascii"
    ).ljust(80)
    assert edf._local_recording_identification == local_recording_identification.encode(
        "ascii"
    ).ljust(80)
    assert edf.__startdate == b"24.04.89"
    assert edf.__starttime == b"16.13.00"
    assert edf._bytes_in_header_record == b"512".ljust(8)
    assert edf._reserved == b"".ljust(44)
    assert edf.__data_record_duration == b"2.5".ljust(8)
    assert edf.__num_signals == b"1".ljust(4)


def test_edf_init_signals_with_different_durations():
    signals = [
        EdfSignal(sine(6, 2, 20), 20, physical_range=(-1, 1)),
        EdfSignal(sine(5, 2, 20), 20, physical_range=(-1, 1)),
    ]
    with pytest.raises(ValueError, match="Inconsistent signal durations"):
        Edf(signals)


@pytest.mark.parametrize("data_record_duration", [1, 2.5, 5, 10, 20])
def test_data_record_duration_exactly_divides_signal_duration(
    data_record_duration: float,
):
    signals = [EdfSignal(sine(20, 2, 30), 30, physical_range=(-1, 1))]
    Edf(signals, data_record_duration=data_record_duration)


@pytest.mark.parametrize("data_record_duration", [1.5, 21, 40])
def test_data_record_duration_does_not_exactly_divide_signal_duration(
    data_record_duration: float,
):
    signals = [EdfSignal(sine(20, 2, 30), 30, physical_range=(-1, 1))]
    with pytest.raises(ValueError, match="Signal duration .* not exactly divisible"):
        Edf(signals, data_record_duration=data_record_duration)


def test_read_write_roundtrip(tmp_file: Path):
    edf = read_edf(EDF_FILE)
    edf.write(tmp_file)
    assert tmp_file.read_bytes() == EDF_FILE.read_bytes()


@pytest.mark.parametrize("data_record_duration", [0.5, 1, 2.5, 5])
def test_write_read_roundtrip(tmp_file: Path, data_record_duration: float):
    signals = [
        EdfSignal(sine(5, 2, 30), 30, physical_range=(-1, 1), label="sine fs=30Hz"),
        EdfSignal(sine(5, 2, 20), 20, physical_range=(-1, 1), label="sine fs=20Hz"),
    ]
    edf = Edf(signals, data_record_duration=data_record_duration)
    edf.write(tmp_file)

    loaded_edf = read_edf(tmp_file)
    assert len(loaded_edf.signals) == len(signals)
    for loaded_signal, reference_signal in zip(loaded_edf.signals, signals):
        assert loaded_signal.sampling_frequency == reference_signal.sampling_frequency
    assert edf.data_record_duration == data_record_duration


@pytest.mark.parametrize("signal_scale", [0.000001, 0.001, 1, 1000, 1000000])
def test_write_read_roundtrip_signal_tolerance(tmp_file: Path, signal_scale: float):
    signals = [
        EdfSignal(
            sine(1, 5, 3000) * signal_scale,
            sampling_frequency=30,
            physical_range=(-signal_scale, signal_scale),
            digital_range=(-32768, 32767),
        ),
    ]
    edf = Edf(signals)
    edf.write(tmp_file)

    loaded_edf = read_edf(tmp_file)
    assert len(loaded_edf.signals) == len(signals)
    for loaded_signal, reference_signal in zip(loaded_edf.signals, signals):
        tolerance = np.ptp(reference_signal.data) * 2**-16
        np.testing.assert_allclose(
            loaded_signal.data,
            reference_signal.data,
            atol=tolerance,
        )


def test_physical_values_are_mapped_to_digital_values_with_minimal_error(
    tmp_file: Path,
):
    data = [3.0, 5.49, 8.5, 1.51, -2.51, -4.5, -2.49]
    expected_result = np.round(data)
    signals = [
        EdfSignal(
            data=np.asarray(data, dtype="float64"),
            sampling_frequency=1,
            physical_range=_FloatRange(-32768, 32767),  # same as digital range
        )
    ]
    Edf(signals).write(tmp_file)
    read_back_edf = read_edf(tmp_file)
    np.testing.assert_array_equal(read_back_edf.signals[0].data, expected_result)


def test_edf_labels_property():
    edf = read_edf(EDF_FILE)
    with EDF_SIGNAL_REFERENCE_FILE.open() as jf:
        reference_labels = tuple(s["label"] for s in json.load(jf))
    assert edf.labels == reference_labels


def test_edf_get_signal_valid_label():
    edf = read_edf(EDF_FILE)
    label = "EMG submental"
    signal = edf.get_signal(label)
    assert signal.label == label
    assert signal.sampling_frequency == 1
    assert signal.physical_min == -5
    assert signal.physical_max == 5


def test_edf_get_signal_nonexistent_label():
    edf = read_edf(EDF_FILE)
    with pytest.raises(ValueError, match="No signal with label"):
        edf.get_signal("some nonexistent label")


def test_edf_get_signal_ambiguous_label():
    label = "Flow Patient"
    edf = Edf(
        [
            EdfSignal(np.arange(10), 1, label=label),
            EdfSignal(np.arange(10), 1, label=label),
        ]
    )
    with pytest.raises(ValueError, match="Ambiguous label"):
        edf.get_signal(label)


@pytest.mark.parametrize(
    "field_name",
    [
        "version",
        "bytes_in_header_record",
        "reserved",
        "num_data_records",
        "num_signals",
    ],
)
def test_edf_field_cannot_be_set_publicly(field_name: str):
    edf = read_edf(EDF_FILE)
    with pytest.raises(AttributeError, match="can't set attribute|has no setter"):
        setattr(edf, field_name, None)


def test_edf_signal_requiring_rounding_in_physical_max_is_written_and_read_correctly(
    tmp_file: Path,
):
    signal = EdfSignal(np.arange(10), 1, physical_range=(0, 10.123456789))
    edf = Edf(signals=[signal])
    edf.write(tmp_file)
    loaded_signal = read_edf(tmp_file).signals[0]
    np.testing.assert_equal(loaded_signal.data, signal.data)


def test_edf_anonymized(tmp_file: Path):
    edf = read_edf(EDF_FILE)
    edf_anon = edf.copy()
    edf_anon.anonymize()
    assert edf_anon.local_patient_identification == "X X X X"
    assert edf.local_patient_identification == "X F X Female_33yr"
    assert edf_anon.local_recording_identification == "Startdate X X X X"
    assert edf.local_recording_identification == "Startdate 24-APR-1989 X X X"
    edf_anon.write(tmp_file)
    assert tmp_file.read_bytes() == EDF_FILE.read_bytes().replace(
        b"X F X Female_33yr",
        b"X X X X          ",
    ).replace(
        b"Startdate 24-APR-1989 X X X",
        b"Startdate X X X X          ",
    ).replace(
        b"24.04.89",
        b"01.01.85",
    ).replace(
        b"16.13.00",
        b"00.00.00",
    )


def test_read_edf_str():
    assert read_edf(str(EDF_FILE)).num_signals == 7


def test_read_edf_expanduser():
    with pytest.raises(FileNotFoundError, match="^[^~]*$"):
        read_edf("~/edfio_testfile_that_should_not_exist.edf")


@pytest.mark.parametrize("mode", ["rb", "r+b"])
def test_read_edf_file_object(mode):
    with EDF_FILE.open(mode) as edf_file:
        assert read_edf(edf_file).num_signals == 7


def test_read_edf_bytes():
    assert read_edf(EDF_FILE.read_bytes()).num_signals == 7


def test_read_edf_bytes_io():
    assert read_edf(io.BytesIO(EDF_FILE.read_bytes())).num_signals == 7


def test_read_edf_spooled_temporary_file():
    with tempfile.SpooledTemporaryFile(mode="w+b") as file:
        file.write(EDF_FILE.read_bytes())
        file.seek(0)
        assert read_edf(file).num_signals == 7


def test_read_edf_writable_file_raises_error(tmp_path: Path):
    with pytest.raises(io.UnsupportedOperation, match="read"):
        with (tmp_path / "test.edf").open("wb") as edf_file:
            assert read_edf(edf_file)


def test_write_edf_str(tmp_file: Path):
    read_edf(EDF_FILE).write(str(tmp_file))
    assert tmp_file.read_bytes() == EDF_FILE.read_bytes()


def test_write_edf_file_object(tmp_file: Path):
    with tmp_file.open("wb") as edf_file:
        read_edf(EDF_FILE).write(edf_file)
    assert tmp_file.read_bytes() == EDF_FILE.read_bytes()


def test_write_edf_bytes_io():
    stream = io.BytesIO()
    read_edf(EDF_FILE).write(stream)
    stream.seek(0)
    assert stream.read() == EDF_FILE.read_bytes()


def test_write_and_read_edf_with_non_ascii_characters_in_path(tmp_path: Path):
    tmp_file = tmp_path / "带 Annotation 的 EDF 文件.edf"
    Edf([EdfSignal(np.arange(10), 1)]).write(tmp_file)
    read_edf(tmp_file)


@pytest.mark.parametrize("data_record_duration", [0.1, 0.3, 0.5, 1.5])
def test_write_edf_with_non_integer_seconds_duration(
    tmp_file: Path,
    data_record_duration: float,
):
    edf = Edf([EdfSignal(np.arange(15), 10)], data_record_duration=data_record_duration)
    edf.write(tmp_file)
    assert read_edf(tmp_file).data_record_duration == data_record_duration


def test_edf_to_bytes():
    assert read_edf(EDF_FILE).to_bytes() == EDF_FILE.read_bytes()


def test_edf_drop_signals_by_indices():
    edf = read_edf(EDF_FILE)
    edf_reduced = edf.copy()
    edf_reduced.drop_signals([0, 2, 4, 5])
    assert edf.num_signals == 7
    assert edf_reduced.num_signals == 3
    assert edf_reduced.bytes_in_header_record == 1024
    assert edf_reduced.labels == ("EEG Pz-Oz", "Resp oro-nasal", "Event marker")


def test_edf_drop_signals_by_labels():
    edf = read_edf(EDF_FILE)
    edf.drop_signals(["Event marker", "EMG submental", "EEG Fpz-Cz"])
    assert edf.num_signals == 4
    assert edf.bytes_in_header_record == 1280
    assert edf.labels == (
        "EEG Pz-Oz",
        "EOG horizontal",
        "Resp oro-nasal",
        "Temp rectal",
    )


def test_edf_drop_signals_by_single_label_given_by_string():
    edf = read_edf(EDF_FILE)
    edf.drop_signals("Event marker")
    assert edf.num_signals == 6
    assert edf.bytes_in_header_record == 7 * 256
    assert edf.labels == (
        "EEG Fpz-Cz",
        "EEG Pz-Oz",
        "EOG horizontal",
        "Resp oro-nasal",
        "EMG submental",
        "Temp rectal",
    )


def test_edf_drop_signals_by_indices_and_labels():
    edf = read_edf(EDF_FILE)
    edf.drop_signals(["Event marker", 1, 5, "EMG submental", 0])
    assert edf.num_signals == 2
    assert edf.bytes_in_header_record == 768
    assert edf.labels == ("EOG horizontal", "Resp oro-nasal")


@pytest.mark.parametrize("drop", [[-1], [7], ["invalid-label"], [1, 2, "invalid"]])
def test_edf_drop_signals_invalid_identifier_raises_error(drop: list[int | str]):
    edf = read_edf(EDF_FILE)
    with pytest.raises(ValueError, match="No signal found with index/label"):
        edf.drop_signals(drop)


def test_edf_drop_signals_removes_all_occurences_for_signals_with_identical_labels():
    edf = Edf(
        [
            EdfSignal(np.arange(2), 1, label=label)
            for label in ("Flow", "ECG", "SpO2", "Flow")
        ],
    )
    edf.drop_signals(["Flow"])
    assert edf.labels == ("ECG", "SpO2")


def test_edf_drop_signals_works_if_signal_is_selected_by_both_index_and_label():
    edf = read_edf(EDF_FILE)
    edf.drop_signals(["Event marker", 6])
    assert edf.num_signals == 6


def test_edf_slice_between_seconds():
    edf = Edf(
        [
            EdfSignal(np.arange(32), 8, physical_range=(-32768, 32767)),
            EdfSignal(np.arange(64), 16, physical_range=(-32768, 32767)),
            EdfSignal(np.arange(1024), 256, physical_range=(-32768, 32767)),
        ],
    )
    edf.slice_between_seconds(2, 3)
    for signal in edf.signals:
        assert len(signal._digital) == signal.sampling_frequency
    np.testing.assert_array_equal(edf.signals[0].data, np.arange(16, 24))
    np.testing.assert_array_equal(edf.signals[1].data, np.arange(32, 48))
    np.testing.assert_array_equal(edf.signals[2].data, np.arange(512, 768))


def test_edf_slice_between_seconds_modifies_header_fields():
    edf = Edf(
        [EdfSignal(np.arange(8 * 3600 * 10), 10)],
        recording=Recording(startdate=datetime.date(1999, 12, 31)),
        starttime=datetime.time(22, 33, 44, 55555),
        annotations=(),
    )
    edf.slice_between_seconds(5 * 3600 + 0.1, 6 * 3600 + 0.1)
    assert edf.startdate == datetime.date(2000, 1, 1)
    assert edf.starttime == datetime.time(3, 33, 44, 155555)
    assert edf.num_data_records == 3600


@pytest.mark.parametrize(("start", "stop"), [(2, 4.5), (2.5, 4)])
def test_edf_slice_between_seconds_not_coinciding_with_sample_times_raises_error(
    start: float,
    stop: float,
):
    edf = Edf([EdfSignal(np.zeros(10), 1), EdfSignal(np.zeros(20), 2)])
    with pytest.raises(ValueError, match="is not a sample time of signal"):
        edf.slice_between_seconds(start, stop)


def test_edf_slice_between_seconds_excludes_stop_sample():
    edf = Edf([EdfSignal(np.arange(10), 1, digital_range=(0, 9))])
    edf.slice_between_seconds(2, 5)
    np.testing.assert_array_equal(edf.signals[0].data, np.arange(2, 5))


def test_edf_slice_between_seconds_shifts_annotation_onsets(tmp_file: Path):
    edf = Edf(
        [EdfSignal(np.zeros(10), 1)],
        annotations=(
            EdfAnnotation(3, None, ""),
            EdfAnnotation(4.7, None, ""),
            EdfAnnotation(5.1234567, None, ""),
        ),
    )
    edf.slice_between_seconds(2, 8)
    edf.write(tmp_file)
    edf_sliced = read_edf(tmp_file)
    assert [ann.onset for ann in edf_sliced.annotations] == [1, 2.7, 3.1234567]


def test_edf_slice_between_seconds_does_not_shift_legacy_startdate_when_edfplus_startdate_is_anonymized():
    edf = Edf([EdfSignal(np.zeros(48), 1 / 3600)])
    edf.slice_between_seconds(36 * 3600, 37 * 3600)
    with pytest.raises(AnonymizedDateError, match="startdate is not available"):
        edf.startdate
    assert edf._startdate == datetime.date(1985, 1, 1)


@pytest.mark.parametrize(("start", "stop"), [(-1, 5), (-1, 11), (5, 11)])
def test_edf_slice_between_seconds_times_outside_recording_raise_error(
    start: float,
    stop: float,
):
    edf = Edf([EdfSignal(np.zeros(10), 1)])
    with pytest.raises(ValueError, match="invalid slice time"):
        edf.slice_between_seconds(start, stop)


def test_edf_slice_between_seconds_drops_annotations_starting_outside_limits(
    tmp_file: Path,
):
    edf = Edf(
        [EdfSignal(np.zeros(10), 1)],
        annotations=(
            EdfAnnotation(2, None, ""),
            EdfAnnotation(2.999, None, ""),
            EdfAnnotation(3, 1, "an annotation"),
            EdfAnnotation(5, None, "another annotation"),
            EdfAnnotation(7.999, 2, "the last annotation"),
            EdfAnnotation(8, None, ""),
        ),
    )
    edf.slice_between_seconds(3, 8)
    edf.write(tmp_file)
    edf_sliced = read_edf(tmp_file)
    assert edf_sliced.annotations == (
        EdfAnnotation(0, 1, "an annotation"),
        EdfAnnotation(2, None, "another annotation"),
        EdfAnnotation(4.999, 2, "the last annotation"),
    )


def test_edf_slice_between_seconds_keep_all_annotations(tmp_file: Path):
    edf = Edf(
        [EdfSignal(np.zeros(10), 1)],
        annotations=(
            EdfAnnotation(2, None, ""),
            EdfAnnotation(3, None, ""),
            EdfAnnotation(7, None, ""),
            EdfAnnotation(8, None, ""),
        ),
    )
    edf.slice_between_seconds(3, 7, keep_all_annotations=True)
    edf.write(tmp_file)
    edf_sliced = read_edf(tmp_file)
    assert [a.onset for a in edf_sliced.annotations] == [-1, 0, 4, 5]


def test_edf_slice_between_annotations():
    edf = Edf(
        [EdfSignal(np.zeros(10), 1)],
        annotations=(
            EdfAnnotation(2, None, "task start"),
            EdfAnnotation(7, None, "task end"),
        ),
    )
    edf.slice_between_annotations("task start", "task end")
    assert edf.starttime == datetime.time(0, 0, 2)
    assert edf.duration == 5


def test_edf_slice_between_annotations_keep_all_annotations(tmp_file: Path):
    edf = Edf(
        [EdfSignal(np.zeros(10), 1)],
        annotations=(
            EdfAnnotation(2, None, ""),
            EdfAnnotation(3, None, "task start"),
            EdfAnnotation(7, None, "task end"),
            EdfAnnotation(8, None, ""),
        ),
    )
    edf.slice_between_annotations(
        "task start",
        "task end",
        keep_all_annotations=True,
    )
    edf.write(tmp_file)
    edf_sliced = read_edf(tmp_file)
    assert [a.onset for a in edf_sliced.annotations] == [-1, 0, 4, 5]


@pytest.mark.parametrize(
    ("start_text", "stop_text"),
    [
        ("invalid", "task end"),
        ("task start", "invalid"),
        ("invalid", "also invalid"),
    ],
)
def test_edf_slice_between_annotations_raises_error_for_invalid_text(
    start_text: str,
    stop_text: str,
):
    edf = Edf(
        [EdfSignal(np.zeros(10), 1)],
        annotations=(
            EdfAnnotation(2, None, "task start"),
            EdfAnnotation(7, None, "task end"),
        ),
    )
    with pytest.raises(ValueError, match="No annotation found with text"):
        edf.slice_between_annotations(start_text, stop_text)


def test_edf_slice_between_annotations_raises_error_for_ambiguous_text():
    edf = Edf(
        [EdfSignal(np.zeros(10), 1)],
        annotations=(
            EdfAnnotation(2, None, "task start"),
            EdfAnnotation(3, None, "task start"),
            EdfAnnotation(7, None, "task end"),
        ),
    )
    with pytest.raises(ValueError, match="Ambiguous annotation text"):
        edf.slice_between_annotations("task start", "task end")


def test_edf_slice_between_annotations_works_for_multiple_annotation_signals():
    edf = Edf(
        [
            EdfSignal(np.zeros(10), 1),
            _create_annotations_signal(
                [
                    EdfAnnotation(2, None, "task start"),
                    EdfAnnotation(5, None, "something"),
                ],
                num_data_records=10,
                data_record_duration=1,
            ),
            _create_annotations_signal(
                [
                    EdfAnnotation(4, None, "something else"),
                    EdfAnnotation(8, None, "task end"),
                ],
                num_data_records=10,
                data_record_duration=1,
                with_timestamps=False,
            ),
        ],
    )
    edf.slice_between_annotations("task start", "task end")
    assert edf.annotations == (
        EdfAnnotation(0, None, "task start"),
        EdfAnnotation(2, None, "something else"),
        EdfAnnotation(3, None, "something"),
    )


def test_edf_slice_between_seconds_keeps_annotations_at_subsecond_start():
    edf = Edf(
        [EdfSignal(np.zeros(20), 2)],
        annotations=(
            EdfAnnotation(3, None, ""),
            EdfAnnotation(3.5, None, ""),
            EdfAnnotation(3.7, None, ""),
            EdfAnnotation(4, None, ""),
        ),
    )
    edf.slice_between_seconds(3.5, 5.5)
    assert [ann.onset for ann in edf.annotations] == [0, 0.2, 0.5]


@pytest.mark.parametrize("data_record_duration", [1 / 4, 1 / 2, 2, 4])
def test_writing_edfplus_works_with_data_record_duration_different_from_1(
    data_record_duration: float,
):
    edf = Edf(
        [EdfSignal(np.arange(144), 12)],
        annotations=(),
        data_record_duration=data_record_duration,
    )
    edf.to_bytes()


@pytest.mark.parametrize(
    ("signal_duration", "data_record_duration", "result"),
    [
        (11, 1, 11),
        (1.1, 0.1, 11),
        (1.01, 0.01, 101),
        (1.001, 0.001, 1001),
        (1.0001, 0.0001, 10001),
        (1.00001, 0.00001, 100001),
        (1.000001, 0.000001, 1000001),
        (1.0000001, 0.0000001, 10000001),
        (1, 1 / 3, 3),
        (1, 1 / 333333, 333333),
    ],
)
def test_calculate_num_data_records(
    signal_duration: float,
    data_record_duration: float,
    result: int,
):
    assert _calculate_num_data_records(signal_duration, data_record_duration) == result


@pytest.mark.parametrize(
    "startdate",
    [
        datetime.date(1984, 12, 31),
        datetime.date(2085, 1, 1),
    ],
)
def test_startdate_outside_edf_compatible_range_raises_error(startdate: datetime.date):
    with pytest.raises(ValueError, match="EDF only allows dates from 1985 to 2084"):
        Edf([EdfSignal(np.arange(2), 1)], recording=Recording(startdate=startdate))


def test_edf_with_only_annotations_can_be_written(tmp_file: Path):
    annotations = (
        EdfAnnotation(0, 30, "Sleep Stage W"),
        EdfAnnotation(30, 150, "Sleep Stage N1"),
        EdfAnnotation(180, 60, "Sleep Stage N2"),
    )
    Edf([], annotations=annotations).write(tmp_file)
    edf = read_edf(tmp_file)
    assert edf.bytes_in_header_record == 512
    assert edf.reserved == "EDF+C"
    assert edf.data_record_duration == 0
    assert edf.num_signals == 0
    assert edf._num_signals == 1
    assert edf.num_data_records == 1
    assert edf.annotations == annotations


def test_edf_with_only_annotations_nonzero_data_record_duration():
    with pytest.raises(ValueError, match="Data record duration must be zero"):
        Edf([], annotations=(EdfAnnotation(0, 10, "ann 1"),), data_record_duration=20)


def test_edf_without_signals_or_annotations_cannot_be_created():
    with pytest.raises(ValueError, match="must contain either signals or annotations"):
        Edf([])


def test_get_starttime_from_file_with_reserved_field_indicating_edfplus_but_no_annotations_signal(
    tmp_file: Path,
):
    starttime = datetime.time(22, 33, 44)
    edf = Edf([EdfSignal(np.arange(2), 1)], starttime=starttime)
    edf._reserved = Edf.reserved.encode("EDF+C")
    edf.write(tmp_file)
    assert read_edf(tmp_file).starttime == starttime


def test_drop_signals_keeps_position_of_annotation_signals():
    edf = Edf(
        signals=[
            EdfSignal(np.arange(2), 1, label="EEG 1"),
            EdfSignal(np.arange(2), 1, label="EEG 2"),
            _create_annotations_signal(
                [EdfAnnotation(0, None, "ann 1")],
                num_data_records=2,
                data_record_duration=1,
            ),
            EdfSignal(np.arange(2), 1, label="EEG 3"),
            EdfSignal(np.arange(2), 1, label="EEG 4"),
            _create_annotations_signal(
                [EdfAnnotation(0.25, None, "ann 2")],
                num_data_records=2,
                data_record_duration=1,
                with_timestamps=False,
            ),
            EdfSignal(np.arange(2), 1, label="EEG 5"),
        ],
    )
    edf.drop_signals([0, 3])
    assert edf.labels == ("EEG 2", "EEG 3", "EEG 5")
    assert edf._signals[1].label == "EDF Annotations"
    assert edf._signals[3].label == "EDF Annotations"
    assert edf.annotations == (
        EdfAnnotation(0, None, "ann 1"),
        EdfAnnotation(0.25, None, "ann 2"),
    )


@pytest.mark.parametrize(
    "new_signal",
    [
        EdfSignal(np.arange(100), 10),
        EdfSignal(np.arange(200), 20),
        EdfSignal(np.arange(30), 3),
    ],
)
def test_append_signals_single(new_signal: EdfSignal):
    edf = Edf([EdfSignal(np.arange(100), 10), EdfSignal(np.arange(50), 5)])
    new_signal = EdfSignal(np.arange(1000), 100)
    edf.append_signals(new_signal)
    assert edf.num_signals == 3
    assert edf.signals[2] == new_signal


def test_append_signals_multiple():
    edf = Edf(
        [
            EdfSignal(np.arange(100), 10),
            EdfSignal(np.arange(50), 5),
        ]
    )
    new_signals = [
        EdfSignal(np.arange(100), 10),
        EdfSignal(np.arange(30), 3),
    ]
    edf.append_signals(new_signals)
    assert edf.num_signals == 2 + len(new_signals)
    for actual, expected in zip(new_signals, edf.signals[2:]):
        assert actual == expected


def test_append_signals_appends_after_last_ordinary_signal():
    edf = Edf(
        [
            EdfSignal(np.arange(3), 1, label="S1"),
            EdfSignal(np.arange(3), 1, label="S2"),
            _create_annotations_signal(
                (), data_record_duration=1, num_data_records=3, with_timestamps=True
            ),
            EdfSignal(np.arange(3), 1, label="S3"),
            _create_annotations_signal(
                (), data_record_duration=1, num_data_records=3, with_timestamps=True
            ),
        ]
    )
    edf.append_signals(
        [
            EdfSignal(np.arange(3), 1, label="S4"),
            EdfSignal(np.arange(3), 1, label="S5"),
        ]
    )
    expected = ["S1", "S2", "EDF Annotations", "S3", "S4", "S5", "EDF Annotations"]
    assert [s.label for s in edf._signals] == expected


@pytest.mark.parametrize(
    ("length", "sampling_frequency"),
    [(1001, 10), (999, 10), (1000, 10.001), (1, 0.011)],
)
def test_append_signals_raises_error_on_duration_mismatch(
    length: int,
    sampling_frequency: float,
):
    edf = Edf([EdfSignal(np.arange(1000), 10)])
    with pytest.raises(ValueError, match="Inconsistent signal durations"):
        edf.append_signals(EdfSignal(np.arange(length), sampling_frequency))


@pytest.mark.parametrize(
    ("length", "sampling_frequency"),
    [(10, 0.5), (60, 3), (22, 1.1), (220, 11)],
)
def test_append_signals_raises_error_for_signals_incompatible_with_the_data_record_duration(
    length: int,
    sampling_frequency: float,
):
    edf = Edf([EdfSignal(np.arange(100), 5)], data_record_duration=0.2)
    with pytest.raises(ValueError, match="Not all signal lengths can be split"):
        edf.append_signals([EdfSignal(np.arange(length), sampling_frequency)])


def test_update_data_record_duration():
    edf = Edf([EdfSignal(np.arange(100), 10)])
    edf.update_data_record_duration(0.1)
    assert edf.data_record_duration == 0.1
    assert edf.num_data_records == 100


def test_update_data_record_duration_method_pad_with_zero():
    edf = Edf(
        [EdfSignal(np.arange(100), 10, physical_range=(0, 99), digital_range=(-50, 49))]
    )
    edf.update_data_record_duration(0.3, method="pad")
    assert edf.data_record_duration == 0.3
    assert edf.num_data_records == 34
    assert edf.duration == 10.2
    np.testing.assert_array_equal(
        edf.signals[0].data,
        np.concatenate(
            [
                np.arange(100),
                np.zeros(2),
            ]
        ),
    )


@pytest.mark.parametrize("physical_range", [(5, 104), (-105, -6)])
def test_update_data_record_duration_method_pad_with_physical_minimum(
    physical_range: _FloatRange,
):
    test_data = np.arange(physical_range[0], physical_range[1] + 1)
    edf = Edf(
        [
            EdfSignal(
                test_data, 10, physical_range=physical_range, digital_range=(-50, 49)
            )
        ]
    )
    edf.update_data_record_duration(0.3, method="pad")
    assert edf.data_record_duration == 0.3
    assert edf.num_data_records == 34
    assert edf.duration == 10.2
    np.testing.assert_array_equal(
        edf.signals[0].data,
        np.concatenate(
            [
                test_data,
                np.ones(2) * physical_range[0],
            ]
        ),
    )


def test_update_data_record_duration_method_truncate():
    edf = Edf(
        [EdfSignal(np.arange(100), 10, physical_range=(0, 99), digital_range=(0, 99))]
    )
    edf.update_data_record_duration(0.3, method="truncate")
    assert edf.data_record_duration == 0.3
    assert edf.num_data_records == 33
    assert edf.duration == 9.9
    np.testing.assert_array_equal(
        edf.signals[0].data,
        np.arange(0, 99),
    )


def test_update_data_record_duration_raises_error_if_signal_duration_is_not_exactly_divisible():
    edf = Edf([EdfSignal(np.arange(100), 10)])
    with pytest.raises(
        ValueError,
        match="Signal duration of 10.0s is not exactly divisible by data_record_duration of 0.3s",
    ):
        edf.update_data_record_duration(0.3)


def test_update_data_record_duration_raises_error_if_sampling_rate_incompatible():
    edf = Edf([EdfSignal(np.arange(100), 10), EdfSignal(np.arange(100), 10)])
    with pytest.raises(
        ValueError,
        match="Cannot set data record duration to 0.05: Incompatible sampling frequency 10.* Hz",
    ):
        edf.update_data_record_duration(0.05)


@pytest.mark.parametrize("record_duration", [0, -1])
def test_update_data_record_duration_raises_error_if_data_record_duration_not_positive(
    record_duration: float,
):
    edf = Edf([EdfSignal(np.arange(100), 10)])
    with pytest.raises(ValueError, match="Data record duration must be positive"):
        edf.update_data_record_duration(record_duration)


def test_update_data_record_duration_with_annotations(tmp_file: Path):
    expected_data = np.arange(30)
    expected_annotations = (
        EdfAnnotation(0, None, "ann 1"),
        EdfAnnotation(10, None, "ann 2"),
        EdfAnnotation(20, None, "ann 3"),
    )
    edf = Edf(
        [EdfSignal(expected_data, 10, physical_range=(0, 29), digital_range=(0, 29))],
        annotations=expected_annotations,
    )
    edf.update_data_record_duration(0.3)
    edf.write(tmp_file)
    edf = read_edf(tmp_file)
    assert edf.data_record_duration == 0.3
    assert edf.num_data_records == 10
    np.testing.assert_array_equal(edf.signals[0].data, expected_data)
    assert edf.annotations == expected_annotations


def test_update_data_record_duration_with_multiple_annotations_signals(tmp_file: Path):
    expected_data = np.arange(30)
    expected_annotations1 = (
        EdfAnnotation(0, None, "ann 1"),
        EdfAnnotation(10, None, "ann 2"),
    )
    expected_annotations2 = (EdfAnnotation(20, None, "ann 3"),)
    edf = Edf(
        [
            EdfSignal(expected_data, 10, physical_range=(0, 29), digital_range=(0, 29)),
            _create_annotations_signal(
                expected_annotations1,
                data_record_duration=1,
                num_data_records=3,
                with_timestamps=True,
            ),
            _create_annotations_signal(
                expected_annotations2,
                data_record_duration=1,
                num_data_records=3,
                with_timestamps=False,
            ),
        ],
        data_record_duration=1,
    )
    edf.update_data_record_duration(0.3)
    edf.write(tmp_file)
    edf = read_edf(tmp_file)
    assert edf.data_record_duration == 0.3
    assert edf.num_data_records == 10
    np.testing.assert_array_equal(edf.signals[0].data, expected_data)
    assert edf.annotations == expected_annotations1 + expected_annotations2


def test_update_data_record_duration_annotations_only_raises_error_if_data_record_duration_not_zero():
    edf = Edf(
        [],
        annotations=(EdfAnnotation(0, None, "ann 1"),),
    )
    with pytest.raises(ValueError, match="Data record duration must be zero"):
        edf.update_data_record_duration(0.3)


def test_update_data_record_duration_annotations_only(tmp_file: Path):
    edf = Edf(
        [],
        annotations=(EdfAnnotation(0, None, "ann 1"),),
    )
    edf.update_data_record_duration(0.0)
    edf.write(tmp_file)
    edf = read_edf(tmp_file)
    assert edf.data_record_duration == 0.0
    assert edf.num_data_records == 1


@pytest.mark.parametrize(
    ("method", "expected_num_records"), [("pad", 14), ("truncate", 13)]
)
def test_update_data_record_duration_pad_or_truncate_with_annotations(
    method: Literal["pad", "truncate"], expected_num_records: int, tmp_file: Path
):
    expected_data = np.arange(40)
    expected_annotations = (
        EdfAnnotation(0, 5, "ann 1"),
        EdfAnnotation(10, 4, "ann 2"),
        EdfAnnotation(20, 10, "ann 3"),
    )
    edf = Edf(
        [EdfSignal(expected_data, 10, physical_range=(0, 39), digital_range=(0, 39))],
        annotations=expected_annotations,
    )
    edf.update_data_record_duration(0.3, method=method)
    edf.write(tmp_file)
    edf = read_edf(tmp_file)
    assert edf.data_record_duration == 0.3
    assert edf.num_data_records == expected_num_records
    assert edf.duration == expected_num_records * 0.3
    assert edf.annotations == expected_annotations


def test_update_data_record_duration_with_subsecond_offset(tmp_file: Path):
    expected_annotations = (EdfAnnotation(5, 5, "ann 1"),)
    microseconds_offset = 153781
    edf = Edf(
        [EdfSignal(np.arange(100), 10, physical_range=(0, 99), digital_range=(0, 99))],
        annotations=expected_annotations,
        starttime=datetime.time(0, 0, 0, microseconds_offset),
    )
    edf.update_data_record_duration(2.5)
    edf.write(tmp_file)
    edf = read_edf(tmp_file)
    assert edf.data_record_duration == 2.5
    assert edf.num_data_records == 4
    assert edf.duration == 10
    assert edf.annotations == expected_annotations
    assert edf.starttime.microsecond == microseconds_offset


def test_sampling_frequencies_leading_to_floating_point_issues_in_signal_duration_calculation(
    tmp_file: Path,
):
    Edf(
        signals=[EdfSignal(np.arange(220), 22 / 9), EdfSignal(np.arange(90), 1)],
        data_record_duration=9,
    ).write(tmp_file)
    edf = read_edf(tmp_file)
    assert edf.num_data_records == 10
    assert edf.signals[0].samples_per_data_record == 22
    assert edf.signals[1].samples_per_data_record == 9


# fmt: off
@pytest.mark.parametrize(
    ("extra_bytes", "num_records_in_header", "expected_warning"),
    [
    #    extra bytes     num records field    expected warning
        (1,              10,                  "Incomplete data record at the end of the EDF file"),
        (15,             11,                  "Incomplete data record at the end of the EDF file"),
        (0,              9,                   "EDF header indicates 9 data records, but file contains 10 records. Updating header."),
        (0,              11,                  "EDF header indicates 11 data records, but file contains 10 records. Updating header."),
    ],
)
# fmt: on
def test_read_edf_with_invalid_number_of_records(
    tmp_path: Path,
    extra_bytes: int,
    num_records_in_header: int,
    expected_warning: str,
):
    invalid_edf = tmp_path / "invalid.edf"
    copyfile(EDF_FILE, invalid_edf)
    with invalid_edf.open("rb+") as file:
        file.seek(236)
        file.write(f"{num_records_in_header: <8}".encode("ascii"))
        if extra_bytes > 0:
            file.seek(0, 2)
            file.write(b"\0" * extra_bytes)

    for io_obj in (invalid_edf, invalid_edf.read_bytes()):
        with pytest.warns(UserWarning, match=expected_warning):
            edf = read_edf(io_obj)
        assert edf.num_data_records == 10
        comparison_edf = read_edf(EDF_FILE)
        for signal, comparison_signal in zip(edf.signals, comparison_edf.signals):
            np.testing.assert_array_equal(signal.data, comparison_signal.data)

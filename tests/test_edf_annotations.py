from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

from edfio import (
    Edf,
    EdfAnnotation,
    EdfSignal,
    read_edf,
)
from edfio.edf_annotations import (
    _EdfAnnotationsDataRecord,
    _encode_annotation_duration,
    _encode_annotation_onset,
)
from edfio.edf_signal import EdfAnnotationsSignal
from tests import TEST_DATA_DIR

MNE_TEST_FILE = TEST_DATA_DIR / "mne_test.edf"
SUBSECOND_TEST_FILE = TEST_DATA_DIR / "test_subsecond.edf"

MNE_TEST_ANNOTATIONS = (
    EdfAnnotation(onset=0.0, duration=None, text="start"),
    EdfAnnotation(onset=0.1344, duration=0.256, text="type A"),
    EdfAnnotation(onset=0.3904, duration=1.0, text="type A"),
    EdfAnnotation(onset=2.0, duration=None, text="type B"),
    EdfAnnotation(onset=2.5, duration=2.5, text="type A"),
)


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
    assert _encode_annotation_onset(onset) == expected


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
    assert _encode_annotation_duration(duration) == expected


def test_encode_annotation_duration_raises_error_for_negative_values():
    with pytest.raises(ValueError, match="Annotation duration must be positive, is"):
        _encode_annotation_duration(-1)


def test_edf_annotations():
    edf = read_edf(MNE_TEST_FILE)
    assert edf.annotations == MNE_TEST_ANNOTATIONS


# examples taken from https://www.edfplus.info/specs/edfplus.html
@pytest.mark.parametrize(
    ("raw_annotations", "expected_annotations"),
    [
        (
            b"+180\x14Lights off\x14Close door\x14\x00",
            [
                EdfAnnotation(180, None, "Lights off"),
                EdfAnnotation(180, None, "Close door"),
            ],
        ),
        (
            b"+180\x14Lights off\x14\x00+180\x14Close door\x14\x00",
            [
                EdfAnnotation(180, None, "Lights off"),
                EdfAnnotation(180, None, "Close door"),
            ],
        ),
        (
            b"+1800.2\x1525.5\x14Apnea\x14\x00",
            [EdfAnnotation(1800.2, 25.5, "Apnea")],
        ),
        (
            b"+0\x14\x14Stimulus click 35dB both ears\x14Free text\x14\x00",
            [
                EdfAnnotation(0, None, ""),
                EdfAnnotation(0, None, "Stimulus click 35dB both ears"),
                EdfAnnotation(0, None, "Free text"),
            ],
        ),
        (
            b"-0.065\x14Pre-stimulus beep 1000Hz\x14\x00",
            [EdfAnnotation(-0.065, None, "Pre-stimulus beep 1000Hz")],
        ),
        (
            b"+0\x14\x14Recording starts\x14\x00",
            [
                EdfAnnotation(0, None, ""),
                EdfAnnotation(0, None, "Recording starts"),
            ],
        ),
        (
            b"+993.2\x151.2\x14Limb movement\x14R+L leg\x14\x00",
            [
                EdfAnnotation(993.2, 1.2, "Limb movement"),
                EdfAnnotation(993.2, 1.2, "R+L leg"),
            ],
        ),
        (
            b"+30210\x14Recording ends\x14\x00\x00\x00\x00\x00\x00\x00\x00",
            [EdfAnnotation(30210, None, "Recording ends")],
        ),
        (  # from https://github.com/mne-tools/mne-testing-data/blob/14a4cbc0ca9b3f268885d91ff058c18737487865/EDF/test_utf8_annotations.edf
            b"+2\x150.500000\x14\xe4\xbb\xb0\xe5\x8d\xa7\x14\x00",
            [EdfAnnotation(2, 0.5, "仰卧")],
        ),
        (  # allow trailing 0
            b"+15.0\x155.0\x14Text\x14\x00",
            [EdfAnnotation(15, 5, "Text")],
        ),
    ],
)
def test_parse_annotations(
    raw_annotations: bytes,
    expected_annotations: list[EdfAnnotation],
):
    assert (
        _EdfAnnotationsDataRecord.from_bytes(raw_annotations).annotations
        == expected_annotations
    )


@pytest.mark.parametrize(
    "raw_annotations",
    [
        b"+15.\x155.5\x14x\x14\x00",
        b"+15.5\x155.\x14x\x14\x00",
    ],
)
def test_parse_annotations_raises_error_for_invalid_timing(raw_annotations: bytes):
    with pytest.raises(ValueError, match="No valid annotations found in"):
        _EdfAnnotationsDataRecord.from_bytes(raw_annotations).annotations


def test_annotations_are_empty_for_legacy_edf():
    assert read_edf(TEST_DATA_DIR / "short_psg.edf").annotations == ()


def test_read_write_does_not_change_edfplus_with_annotations(tmp_file: Path):
    source_path = MNE_TEST_FILE
    read_edf(source_path).write(tmp_file)
    assert tmp_file.read_bytes() == source_path.read_bytes()


def test_write_edf_with_annotations(tmp_file: Path):
    annotations = (EdfAnnotation(3.5, 3, text="huhu"), EdfAnnotation(5, 1, "仰卧"))
    Edf([EdfSignal(np.arange(10), 1)], annotations=annotations).write(tmp_file)
    edf = read_edf(tmp_file)
    assert edf.annotations == annotations
    assert edf.reserved == "EDF+C"
    assert edf.num_signals == 1
    assert edf._num_signals == 2


def test_annotations_outside_recording_limits(tmp_file: Path):
    annotations = (
        EdfAnnotation(-1, 1, "before start"),
        EdfAnnotation(0, 1, "at start"),
        EdfAnnotation(1, 1, "at end"),
        EdfAnnotation(2, 1, "after end"),
    )
    edf = Edf(
        signals=[EdfSignal(np.arange(2), sampling_frequency=1)],
        annotations=annotations,
    )
    assert edf.annotations == annotations


def test_drop_annotations():
    edf = read_edf(MNE_TEST_FILE)
    edf.drop_annotations(text="type A")
    expected_annotations = (
        EdfAnnotation(onset=0, duration=None, text="start"),
        EdfAnnotation(onset=2, duration=None, text="type B"),
    )
    assert edf.annotations == expected_annotations


def test_drop_annotations_write(tmp_file: Path):
    edf = read_edf(MNE_TEST_FILE)
    edf.drop_annotations(text="type A")
    edf.write(tmp_file)
    expected_annotations = tuple(a for a in edf.annotations if a.text != "type A")
    assert read_edf(tmp_file).annotations == expected_annotations


def test_drop_annotations_with_instantiated_edf():
    edf = Edf(
        [
            EdfSignal(np.zeros(10), 1),
        ],
        annotations=[
            EdfAnnotation(1, None, "A"),
            EdfAnnotation(2, None, "B"),
        ],
    )
    edf.drop_annotations("A")
    assert edf.annotations == (EdfAnnotation(2, None, "B"),)


def test_annotations_without_text_are_kept(tmp_file: Path):
    annotations = (
        EdfAnnotation(0, None, "abc"),
        EdfAnnotation(1, 1, ""),
    )
    edf = Edf(
        signals=[EdfSignal(np.arange(2), sampling_frequency=1)],
        annotations=annotations,
    )
    edf.write(tmp_file)
    assert read_edf(tmp_file).annotations == annotations


def test_read_subsecond_starttime():
    # testfile from PyEDFlib
    edf = read_edf(SUBSECOND_TEST_FILE)
    assert edf.starttime == datetime.time(4, 5, 56, 394531)


def test_write_read_subsecond_starttime(tmp_file: Path):
    starttime = datetime.time(14, 42, 30, 12345)
    edf = Edf(
        signals=[EdfSignal(np.arange(2), sampling_frequency=1)],
        starttime=starttime,
        annotations=(),
    )
    edf.write(tmp_file)
    assert read_edf(tmp_file).starttime == starttime


def test_edf_anonymized_removes_starttime_microseconds():
    edf = read_edf(SUBSECOND_TEST_FILE)
    edf.anonymize()
    assert edf.starttime == datetime.time(0, 0, 0)


def test_edf_write_works_after_adding_microsecond_places(tmp_file: Path):
    edf = Edf(
        signals=[EdfSignal(np.arange(2), sampling_frequency=1)],
        starttime=datetime.time(0, 0, 0),
        annotations=(),
    )
    starttime = datetime.time(0, 0, 0, 123456)
    edf.starttime = starttime
    edf.write(tmp_file)
    assert read_edf(tmp_file).starttime == starttime


def test_edf_annotations_multiple_annotation_signals():
    edf = Edf(
        signals=[
            EdfSignal(np.arange(2), sampling_frequency=1),
            EdfAnnotationsSignal(
                [
                    EdfAnnotation(0, None, "sig 1 ann 1"),
                    EdfAnnotation(0.5, None, "sig 1 ann 2"),
                ],
                num_data_records=2,
                data_record_duration=1,
            ),
            EdfAnnotationsSignal(
                [
                    EdfAnnotation(0.25, None, "sig 2 ann 1"),
                    EdfAnnotation(0.75, None, "sig 2 ann 2"),
                ],
                num_data_records=2,
                data_record_duration=1,
                with_timestamps=False,
            ),
        ],
    )
    assert edf.annotations == (
        EdfAnnotation(0, None, "sig 1 ann 1"),
        EdfAnnotation(0.25, None, "sig 2 ann 1"),
        EdfAnnotation(0.5, None, "sig 1 ann 2"),
        EdfAnnotation(0.75, None, "sig 2 ann 2"),
    )


TEST_SUBSECOND_ANNOTATION_ONSETS = [1.9511719, 3.4921875, 290.5019531, 583.5722656]


def test_annotation_onsets_are_correct_for_file_with_subsecond_starttime_offset():
    edf = read_edf(SUBSECOND_TEST_FILE)
    assert [ann.onset for ann in edf.annotations] == TEST_SUBSECOND_ANNOTATION_ONSETS


def test_changing_starttime_microseconds_does_not_shift_annotation_onsets(
    tmp_file: Path,
):
    edf = read_edf(SUBSECOND_TEST_FILE)
    edf.starttime = edf.starttime.replace(microsecond=1)
    edf.write(tmp_file)
    edf = read_edf(tmp_file)
    assert [ann.onset for ann in edf.annotations] == TEST_SUBSECOND_ANNOTATION_ONSETS


def test_annotation_onsets_are_written_correctly_for_new_edf_with_microsecond_starttime(
    tmp_file: Path,
):
    annotations = (
        EdfAnnotation(-2, None, ""),
        EdfAnnotation(0.2, None, ""),
        EdfAnnotation(1.2345, None, ""),
        EdfAnnotation(15.42, None, ""),
    )
    edf = Edf(
        [EdfSignal(np.arange(10), 1)],
        starttime=datetime.time(0, 0, 0, 123456),
        annotations=annotations,
    )
    edf.write(tmp_file)
    assert read_edf(tmp_file).annotations == annotations


def test_creating_edf_with_microsecond_starttime_without_annotations_emits_warning():
    with pytest.warns(UserWarning, match="Creating EDF\\+C to store microsecond"):
        edf = Edf([EdfSignal(np.arange(2), 1)], starttime=datetime.time(0, 0, 0, 1))
    assert edf.starttime.microsecond == 1
    assert edf.reserved == "EDF+C"


def test_read_edf_containing_only_annotations():
    # testfile from
    # https://www.physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf
    edf = read_edf(TEST_DATA_DIR / "SC4001EC-Hypnogram.edf")
    assert len(edf.annotations) == 154


def test_long_annotation_texts_are_possible(tmp_file: Path):
    annotations = (
        EdfAnnotation(0.1, None, "this is a short text"),
        EdfAnnotation(0.2, None, "this is a looooooooooooooooooooooooooooooooong text"),
        EdfAnnotation(0.3, None, "x" * 10000),
    )
    Edf([EdfSignal(np.arange(10), 1)], annotations=annotations).write(tmp_file)
    assert read_edf(tmp_file).annotations == annotations


def test_high_numbers_of_annotations_are_possible(tmp_file: Path):
    fs = 1000
    signal = EdfSignal(np.arange(fs), fs)
    annotations = (EdfAnnotation(t / fs, 1 / fs, "x") for t in range(fs))
    edf = Edf([signal], annotations=annotations)
    edf.write(tmp_file)
    assert len(read_edf(tmp_file).annotations) == fs


def test_starttime_raises_helpful_error_for_invalid_timestamp_annotation():
    edf = Edf(
        [
            EdfSignal(np.arange(1), 1),
            EdfAnnotationsSignal(
                [EdfAnnotation(2.345, None, "")],
                num_data_records=1,
                data_record_duration=1,
                with_timestamps=False,
            ),
        ]
    )
    edf._reserved = Edf.reserved.encode("EDF+C")
    with pytest.raises(ValueError, match="Subsecond offset in first annotation must"):
        edf.starttime


def test_edf_anonymized_does_not_remove_annotations():
    edf = read_edf(MNE_TEST_FILE)
    edf.anonymize()
    assert edf.annotations == MNE_TEST_ANNOTATIONS

from __future__ import annotations

import contextlib
import copy
import datetime
import io
import math
import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import singledispatch
from math import ceil, floor
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from edfio._header_field import (
    RawHeaderFieldDate,
    RawHeaderFieldFloat,
    RawHeaderFieldInt,
    RawHeaderFieldStr,
    RawHeaderFieldTime,
    encode_str,
    get_header_fields,
)
from edfio._utils import (
    FloatRange,
    IntRange,
    calculate_gain_and_offset,
    decode_edfplus_date,
    encode_annotation_duration,
    encode_annotation_onset,
    encode_edfplus_date,
    repr_from_init,
    round_float_to_8_characters,
    validate_subfields,
)

_ANNOTATIONS_PATTERN = re.compile(
    """
    ([+-]\\d+(?:\\.?\\d+)?)       # onset
    (?:\x15(\\d+(?:\\.?\\d+)?))?  # duration, optional
    (?:\x14(.*?))                 # annotation texts
    \x14\x00                      # terminator
    """,
    re.VERBOSE,
)


class AnonymizedDateError(ValueError):
    """Raised when trying to access an anonymized startdate or birthdate."""


class EdfAnnotation(NamedTuple):
    """A single EDF+ annotation.

    Parameters
    ----------
    onset : float
        The annotation onset in seconds from recording start.
    duration : float | None
        The annotation duration in seconds (`None` if annotation has no duration).
    text : str
        The annotation text, can be empty.
    """

    onset: float
    duration: float | None
    text: str


class EdfSignal:
    """A single EDF signal.

    Attributes that might break the signal or file on modification (i.e.,
    `sampling_frequency`, `physical_range`, `digital_range`, `samples_per_data_record`,
    and `reserved`) can not be set after instantiation.

    To reduce memory consumption, signal data is always stored as a 16-bit integer array
    containing the digital values that would be written to the corresponding EDF file.
    Therefore, it is expected that `EdfSignal.data` does not match the physical
    values passed during instantiation exactly.

    Parameters
    ----------
    data : npt.NDArray[np.float64]
        The signal data (physical values).
    sampling_frequency : float
        The sampling frequency in Hz.
    label : str, default: `""`
        The signal's label, e.g., `"EEG Fpz-Cz"` or `"Body temp"`.
    transducer_type : str, default: `""`
        The transducer type, e.g., `"AgAgCl electrode"`.
    physical_dimension : str, default: `""`
        The physical dimension, e.g., `"uV"` or `"degreeC"`
    physical_range : tuple[float, float] | None, default: None
        The physical range given as a tuple of `(physical_min, physical_max)`. If
        `None`, this is determined from the data.
    digital_range : tuple[int, int], default: `(-32768, 32767)`
        The digital range given as a tuple of `(digital_min, digital_max)`. Uses the
        maximum resolution of 16-bit integers by default.
    prefiltering : str, default: `""`
        The signal prefiltering, e.g., `"HP:0.1Hz LP:75Hz"`.
    """

    label = RawHeaderFieldStr(16, is_settable=True)
    """Signal label, e.g., `"EEG Fpz-Cz"` or `"Body temp"`."""
    transducer_type = RawHeaderFieldStr(80, is_settable=True)
    """Transducer type, e.g., `"AgAgCl electrode"`."""
    physical_dimension = RawHeaderFieldStr(8, is_settable=True)
    """Physical dimension, e.g., `"uV"` or `"degreeC"`."""
    physical_min = RawHeaderFieldFloat(8)
    """Physical minimum, e.g., `-500` or `34`."""
    physical_max = RawHeaderFieldFloat(8)
    """Physical maximum, e.g., `500` or `40`."""
    digital_min = RawHeaderFieldInt(8)
    """Digital minimum, e.g., `-2048`."""
    digital_max = RawHeaderFieldInt(8)
    """Digital maximum, e.g., `2047`."""
    prefiltering = RawHeaderFieldStr(80, is_settable=True)
    """Signal prefiltering, e.g., `"HP:0.1Hz LP:75Hz"`."""
    samples_per_data_record = RawHeaderFieldInt(8)
    """
    Number of samples in each data record.

    For newly instantiated :class:`EdfSignal` objects, this is only set once
    :meth:`Edf.write` is called.
    """
    reserved = RawHeaderFieldStr(32)
    """Reserved signal header field, always `""`"""

    def __init__(
        self,
        data: npt.NDArray[np.float64],
        sampling_frequency: float,
        *,
        label: str = "",
        transducer_type: str = "",
        physical_dimension: str = "",
        physical_range: tuple[float, float] | None = None,
        digital_range: tuple[int, int] = (-32768, 32767),
        prefiltering: str = "",
    ):
        self._sampling_frequency = sampling_frequency
        self.label = label
        self.transducer_type = transducer_type
        self.physical_dimension = physical_dimension
        self.prefiltering = prefiltering
        self._reserved = EdfSignal.reserved.encode("")
        if not np.all(np.isfinite(data)):
            raise ValueError("Signal data must contain only finite values")
        self._set_physical_range(physical_range, data)
        self._set_digital_range(digital_range)
        self._set_data(data)

    def __repr__(self) -> str:
        return repr_from_init(self)

    @classmethod
    def _from_raw_header(
        cls,
        sampling_frequency: float,
        *,
        label: bytes,
        transducer_type: bytes,
        physical_dimension: bytes,
        physical_min: bytes,
        physical_max: bytes,
        digital_min: bytes,
        digital_max: bytes,
        prefiltering: bytes,
        samples_per_data_record: bytes,
        reserved: bytes,
    ) -> EdfSignal:
        sig = object.__new__(cls)
        sig._sampling_frequency = sampling_frequency
        sig._label = label  # type: ignore[attr-defined]
        sig._transducer_type = transducer_type  # type: ignore[attr-defined]
        sig._physical_dimension = physical_dimension  # type: ignore[attr-defined]
        sig._physical_min = physical_min  # type: ignore[attr-defined]
        sig._physical_max = physical_max  # type: ignore[attr-defined]
        sig._digital_min = digital_min  # type: ignore[attr-defined]
        sig._digital_max = digital_max  # type: ignore[attr-defined]
        sig._prefiltering = prefiltering  # type: ignore[attr-defined]
        sig._samples_per_data_record = samples_per_data_record  # type: ignore[attr-defined]
        sig._reserved = reserved  # type: ignore[attr-defined]
        return sig

    @classmethod
    def from_hypnogram(
        cls,
        stages: npt.NDArray[np.float64],
        stage_duration: float = 30,
        *,
        label: str = "",
    ) -> EdfSignal:
        """Create an EDF signal from a hypnogram, with scaling according to EDF specs.

        According to the EDF FAQ [1]_, use integer numbers 0, 1, 2, 3, 4, 5, 6, and 9
        for sleep stages W, 1, 2, 3, 4, R, MT, und unscored, respectively. The digital
        range is set to `(0, 9)`.

        Parameters
        ----------
        stages : npt.NDArray[np.float64]
            The sleep stages, coded as integer numbers.
        stage_duration : float, default: `30`
            The duration of each sleep stage in seconds, used to set the sampling
            frequency to its inverse.
        label : str, default: `""`
            The signal's label.

        Returns
        -------
        EdfSignal
            The resulting :class:`EdfSignal` object.

        References
        ----------
        .. [1] EDF FAQ, https://www.edfplus.info/specs/edffaq.html
        """
        allowed_stages = {0, 1, 2, 3, 4, 5, 6, 9}
        if invalid_stages := set(stages) - allowed_stages:
            raise ValueError(f"stages contains invalid values: {invalid_stages}")
        return EdfSignal(
            data=stages,
            sampling_frequency=1 / stage_duration,
            label=label,
            physical_range=(0, 9),
            digital_range=(0, 9),
        )

    @property
    def physical_range(self) -> FloatRange:
        """The physical range as a tuple of `(physical_min, physical_max)`."""
        return FloatRange(self.physical_min, self.physical_max)

    @property
    def digital_range(self) -> IntRange:
        """The digital range as a tuple of `(digital_min, digital_max)`."""
        return IntRange(self.digital_min, self.digital_max)

    @property
    def sampling_frequency(self) -> float:
        """The sampling frequency in Hz."""
        return self._sampling_frequency

    @property
    def data(self) -> npt.NDArray[np.float64]:
        """
        Numpy array containing the physical signal values as floats.

        To simplify avoiding inconsistencies between signal data and header fields,
        individual values in the returned array can not be modified. Use
        :meth:`EdfSignal.update_data` to overwrite with new physical data.
        """
        try:
            gain, offset = calculate_gain_and_offset(
                self.digital_min,
                self.digital_max,
                self.physical_min,
                self.physical_max,
            )
        except ZeroDivisionError:
            data = self._digital.astype(np.float64)
            warnings.warn(
                f"Digital minimum equals digital maximum ({self.digital_min}) for {self.label}, returning uncalibrated signal."
            )
        except ValueError:
            data = self._digital.astype(np.float64)
        else:
            data = (self._digital + offset) * gain
        data.setflags(write=False)
        return data

    def update_data(
        self,
        data: npt.NDArray[np.float64],
        *,
        keep_physical_range: bool = False,
        sampling_frequency: float | None = None,
    ) -> None:
        """
        Overwrite physical signal values with an array of equal length.

        Parameters
        ----------
        data : npt.NDArray[np.float64]
            The new physical data.
        keep_physical_range : bool, default: False
            If `True`, the `physical_range` is not modified to accomodate the new data.
        sampling_frequency : float | None, default: None
            If not `None`, the `sampling_frequency` is updated to the new value. The new
            data must match the expected length for the new sampling frequency.
        """
        expected_length = len(self._digital)
        if (
            sampling_frequency is not None
            and sampling_frequency != self._sampling_frequency
        ):
            expected_length = self._get_expected_new_length(sampling_frequency)
        if len(data) != expected_length:
            raise ValueError(
                f"Signal lengths must match: got {len(data)}, expected {len(self._digital)}."
            )
        physical_range = self.physical_range if keep_physical_range else None
        self._set_physical_range(physical_range, data)
        if sampling_frequency is not None:
            self._sampling_frequency = sampling_frequency
        self._set_data(data)

    def _get_expected_new_length(self, sampling_frequency: float) -> int:
        if sampling_frequency <= 0:
            raise ValueError(
                f"Sampling frequency must be positive, got {sampling_frequency}"
            )
        current_length = len(self._digital)
        expected_length_f = (
            sampling_frequency / self._sampling_frequency * current_length
        )
        relative_tolerance = 1e-10
        if (
            abs(expected_length_f - round(expected_length_f)) / expected_length_f
            > relative_tolerance
        ):
            raise ValueError(
                f"Sampling frequency of {sampling_frequency} results in non-integer number of samples ({expected_length_f})"
            )
        return round(expected_length_f)

    def _set_digital_range(self, digital_range: tuple[int, int]) -> None:
        digital_range = IntRange(*digital_range)
        if digital_range.min == digital_range.max:
            raise ValueError(
                f"Digital minimum ({digital_range.min}) must differ from digital maximum ({digital_range.max})."
            )
        self._digital_min = EdfSignal.digital_min.encode(digital_range.min)
        self._digital_max = EdfSignal.digital_max.encode(digital_range.max)

    def _set_physical_range(
        self,
        physical_range: tuple[float, float] | None,
        data: npt.NDArray[np.float64],
    ) -> None:
        if physical_range is None:
            physical_range = FloatRange(data.min(), data.max())
            if physical_range.min == physical_range.max:
                physical_range = FloatRange(physical_range.min, physical_range.max + 1)
        else:
            physical_range = FloatRange(*physical_range)
            if physical_range.min == physical_range.max:
                raise ValueError(
                    f"Physical minimum ({physical_range.min}) must differ from physical maximum ({physical_range.max})."
                )
            data_min = data.min()
            data_max = data.max()
            if data_min < physical_range.min or data_max > physical_range.max:
                raise ValueError(
                    f"Signal range [{data_min}, {data_max}] out of physical range: [{physical_range.min}, {physical_range.max}]"
                )
        self._physical_min = EdfSignal.physical_min.encode(
            round_float_to_8_characters(physical_range.min, math.floor)
        )
        self._physical_max = EdfSignal.physical_max.encode(
            round_float_to_8_characters(physical_range.max, math.ceil)
        )

    def _set_data(self, data: npt.NDArray[np.float64]) -> None:
        gain, offset = calculate_gain_and_offset(
            self.digital_min,
            self.digital_max,
            self.physical_min,
            self.physical_max,
        )
        self._digital = np.round(data / gain - offset).astype(np.int16)


class Patient:
    """
    Object representation of the local patient identification.

    Parsing from/to the string containing the local_patient_identification header field
    is done according to the EDF+ specs. Subfields must be ASCII (32..126) and may not
    contain spaces.

    Parameters
    ----------
    code : str, default: `"X"`
        The code by which the patient is known in the hospital administration.
    sex : `{"X", "F", "M"}`, default: `"X"`
        Sex, `F` for female, `M` for male, `X` if anonymized.
    birthdate : datetime.date | None, default: None
        Patient birthdate, stored as `X` if `None`.
    name : str, default: `"X"`
        The patient's name, stored as `X` if `None`.
    additional : Sequence[str], default: `()`
        Optional additional subfields. Will be stored in the header field separated by
        spaces.
    """

    def __init__(
        self,
        *,
        code: str = "X",
        sex: Literal["F", "M", "X"] = "X",
        birthdate: datetime.date | None = None,
        name: str = "X",
        additional: Sequence[str] = (),
    ) -> None:
        if sex not in ("F", "M", "X"):
            raise ValueError(f"Invalid sex: {sex}, must be one of F, M, X")
        if birthdate is None:
            birthdate_field = "X"
        else:
            birthdate_field = encode_edfplus_date(birthdate)
        subfields = {
            "code": code,
            "sex": sex,
            "birthdate": birthdate_field,
            "name": name,
            **{f"additional[{i}]": v for i, v in enumerate(additional)},
        }
        validate_subfields(subfields)
        local_patient_identification = " ".join(subfields.values())
        encode_str(local_patient_identification, 80)
        self._local_patient_identification = local_patient_identification

    def __repr__(self) -> str:
        try:
            return repr_from_init(self)
        except Exception:
            return repr(self._local_patient_identification)

    @classmethod
    def _from_str(cls, string: str) -> Patient:
        encode_str(string, 80)
        obj = object.__new__(cls)
        obj._local_patient_identification = string
        return obj

    def _to_str(self) -> str:
        return self._local_patient_identification

    @property
    def code(self) -> str:
        """The code by which the patient is known in the hospital administration."""
        return self._local_patient_identification.split()[0]

    @property
    def sex(self) -> str:
        """Sex, `F` for female, `M` for male, `X` if anonymized."""
        return self._local_patient_identification.split()[1]

    @property
    def birthdate(self) -> datetime.date:
        """Patient birthdate."""
        birthdate_field = self._local_patient_identification.split()[2]
        if birthdate_field == "X":
            raise AnonymizedDateError("Patient birthdate is not available ('X').")
        return decode_edfplus_date(birthdate_field)

    @property
    def name(self) -> str:
        """The patient's name."""
        return self._local_patient_identification.split()[3]

    @property
    def additional(self) -> tuple[str, ...]:
        """Optional additional subfields."""
        return tuple(self._local_patient_identification.split()[4:])


class Recording:
    """
    Object representation of the local recording identification.

    Parsing from/to the string containing the local_recording_identification header
    field is done according to the EDF+ specs. Subfields must be ASCII (32..126) and may
    not contain spaces.

    Parameters
    ----------
    startdate : datetime.date | None, default: None
        The recording startdate.
    hospital_administration_code : str, default: `"X"`
        The hospital administration code of the investigation, e.g., EEG number or PSG
        number.
    investigator_technician_code : str, default: `"X"`
        A code specifying the responsible investigator or technician.
    equipment_code : str, default: `"X"`
        A code specifying the used equipment.
    additional : Sequence[str], default: `()`
        Optional additional subfields. Will be stored in the header field separated by
        spaces.
    """

    def __init__(
        self,
        *,
        startdate: datetime.date | None = None,
        hospital_administration_code: str = "X",
        investigator_technician_code: str = "X",
        equipment_code: str = "X",
        additional: Sequence[str] = (),
    ) -> None:
        if startdate is None:
            startdate_field = "X"
        else:
            startdate_field = encode_edfplus_date(startdate)
        subfields = {
            "startdate": startdate_field,
            "hospital_administration_code": hospital_administration_code,
            "investigator_technician_code": investigator_technician_code,
            "equipment_code": equipment_code,
            **{f"additional[{i}]": v for i, v in enumerate(additional)},
        }
        validate_subfields(subfields)
        local_recording_identification = " ".join(("Startdate", *subfields.values()))
        encode_str(local_recording_identification, 80)
        self._local_recording_identification = local_recording_identification

    def __repr__(self) -> str:
        try:
            return repr_from_init(self)
        except Exception:
            return repr(self._local_recording_identification)

    @classmethod
    def _from_str(cls, string: str) -> Recording:
        encode_str(string, 80)
        obj = object.__new__(cls)
        obj._local_recording_identification = string
        return obj

    def _to_str(self) -> str:
        return self._local_recording_identification

    @property
    def startdate(self) -> datetime.date:
        """The recording startdate."""
        if not self._local_recording_identification.startswith("Startdate "):
            raise ValueError(
                f"Local recording identification field {self._local_recording_identification!r} does not follow EDF+ standard."
            )
        startdate_field = self._local_recording_identification.split()[1]
        if startdate_field == "X":
            raise AnonymizedDateError("Recording startdate is not available ('X').")
        return decode_edfplus_date(startdate_field)

    @property
    def hospital_administration_code(self) -> str:
        """The hospital administration code of the investigation."""
        return self._local_recording_identification.split()[2]

    @property
    def investigator_technician_code(self) -> str:
        """A code specifying the responsible investigator or technician."""
        return self._local_recording_identification.split()[3]

    @property
    def equipment_code(self) -> str:
        """A code specifying the used equipment."""
        return self._local_recording_identification.split()[4]

    @property
    def additional(self) -> tuple[str, ...]:
        """Optional additional subfields."""
        return tuple(self._local_recording_identification.split()[5:])


class Edf:
    """Python representation of an EDF file.

    EDF header fields are exposed as properties with appropriate data types (i.e.,
    string, numeric, date, or time objects). Fields that might break the file on
    modification (i.e., `version`, `bytes_in_header_record`, `reserved`,
    `num_data_records`, `data_record_duration`, and `num_signals`) can not be set after
    instantiation.

    Note that the startdate has to be set via the parameter `recording`.

    For writing an EDF file with a non-integer seconds duration, currently an
    appropriate value for `data_record_duration` has to be provided manually.

    Parameters
    ----------
    signals : Sequence[EdfSignal]
        The (non-annotation) signals to be contained in the EDF file.
    patient : Patient | None, default: None
        The "local patient identification", containing patient code, sex, birthdate,
        name, and optional additional fields. If `None`, the field is set to `X X X X`
        in accordance with EDF+ specs.
    recording : Recording | None, default: None
        The "local recording identification", containing recording startdate, hospital
        administration code, investigator/technical code, equipment code, and optional
        additional fields. If `None`, the field is set to `Startdate X X X X` in
        accordance with EDF+ specs.
    starttime : datetime.time | None, default: None
        The starttime of the recording. If `None`, `00.00.00` is used. If `starttime`
        contains microseconds, an EDF+C file is created.
    data_record_duration : float | None, default: None
        The duration of each data record in seconds. If `None`, an appropriate value is
        chosen automatically.
    annotations : Iterable[EdfAnnotation] | None, default: None
        The annotations, consisting of onset, duration (optional), and text. If not
        `None`, an EDF+C file is created.
    """

    version = RawHeaderFieldInt(8)
    """EDF version, always `0`"""
    local_patient_identification = RawHeaderFieldStr(80, is_settable=True)
    """
    Unparsed string representation of the legacy local patient identification.

    See also
    --------
    patient: Parsed representation, as a :class:`Patient` object.
    """
    local_recording_identification = RawHeaderFieldStr(80, is_settable=True)
    """
    Unparsed string representation of the legacy local recording identification.

    See also
    --------
    recording: Parsed representation, as a :class:`Recording` object.
    """
    _startdate = RawHeaderFieldDate(8, is_settable=True)
    _starttime = RawHeaderFieldTime(8, is_settable=True)
    bytes_in_header_record = RawHeaderFieldInt(8)
    """Number of bytes in the header record."""
    reserved = RawHeaderFieldStr(44)
    """`"EDF+C"` for an EDF+C file, else `""`."""
    num_data_records = RawHeaderFieldInt(8)
    """Number of data records in the recording."""
    _data_record_duration = RawHeaderFieldFloat(8, is_settable=True)
    num_signals = RawHeaderFieldInt(4)
    """Number of signals in the recording, including annotation signals for EDF+."""

    def __init__(
        self,
        signals: Sequence[EdfSignal],
        *,
        patient: Patient | None = None,
        recording: Recording | None = None,
        starttime: datetime.time | None = None,
        data_record_duration: float | None = None,
        annotations: Iterable[EdfAnnotation] | None = None,
    ):
        if not signals and not annotations:
            raise ValueError("Edf must contain either signals or annotations")
        if patient is None:
            patient = Patient()
        if recording is None:
            recording = Recording()
        if starttime is None:
            starttime = datetime.time(0, 0, 0)
        if data_record_duration is None:
            data_record_duration = _calculate_data_record_duration(signals)
        elif len(signals) == 0 and data_record_duration != 0:
            raise ValueError(
                "Data record duration must be zero for annotation-only files"
            )

        self._data_record_duration = data_record_duration
        self._set_num_data_records_with_signals(signals)
        self._version = Edf.version.encode(0)
        self.local_patient_identification = patient._to_str()
        self.local_recording_identification = recording._to_str()
        self._set_startdate_with_recording(recording)
        self._starttime = starttime.replace(microsecond=0)
        self._reserved = Edf.reserved.encode("")
        if starttime.microsecond and annotations is None:
            warnings.warn("Creating EDF+C to store microsecond starttime.")
        if annotations is not None or starttime.microsecond:
            signals = (
                *signals,
                _create_annotations_signal(
                    annotations if annotations is not None else (),
                    num_data_records=self.num_data_records,
                    data_record_duration=self.data_record_duration,
                    subsecond_offset=starttime.microsecond / 1_000_000,
                ),
            )
            self._reserved = Edf.reserved.encode("EDF+C")
        self._set_signals(signals)

    def __repr__(self) -> str:
        return repr_from_init(self)

    def _load_data(self, file: Path | io.BufferedReader | io.BytesIO) -> None:
        lens = [signal.samples_per_data_record for signal in self.signals]
        datarecord_len = sum(lens)
        if not isinstance(file, Path):
            datarecords = np.frombuffer(file.read(), dtype=np.int16)
        else:
            datarecords = np.memmap(
                file,
                dtype=np.int16,
                mode="r",
                offset=self.bytes_in_header_record,
            )
        datarecords.shape = (self.num_data_records, datarecord_len)
        ends = np.cumsum(lens)
        starts = ends - lens

        for signal, start, end in zip(self.signals, starts, ends):
            signal._digital = datarecords[:, start:end].flatten()

    def _read_header(self, buffer: io.BufferedReader | io.BytesIO) -> None:
        for header_name, length in get_header_fields(Edf):
            setattr(self, "_" + header_name, buffer.read(length))
        self._signals = self._parse_signal_headers(buffer.read(256 * self.num_signals))

    @property
    def signals(self) -> tuple[EdfSignal, ...]:
        """
        All signals contained in the recording, including annotation signals.

        Individual signals can not be removed, added, or replaced by modifying this
        property. Use :meth:`Edf.append_signals`, :meth:`Edf.drop_signals`, or
        :attr:`EdfSignal.data`, respectively.
        """
        return self._signals

    def _set_signals(self, signals: Sequence[EdfSignal]) -> None:
        signals = tuple(signals)
        self._set_num_data_records_with_signals(signals)
        self._signals = signals
        self._bytes_in_header_record = Edf.bytes_in_header_record.encode(
            256 * (len(signals) + 1)
        )
        self._num_signals = Edf.num_signals.encode(len(signals))
        if all(s.label == "EDF Annotations" for s in signals):
            self._data_record_duration = 0

    def _set_num_data_records_with_signals(
        self,
        signals: Sequence[EdfSignal],
    ) -> None:
        if not signals:
            num_data_records = 1
        else:
            signal_durations = [
                round(len(s._digital) / s.sampling_frequency, 12) for s in signals
            ]
            if any(v != signal_durations[0] for v in signal_durations[1:]):
                raise ValueError(
                    f"Inconsistent signal durations (in seconds): {signal_durations}"
                )
            num_data_records = _calculate_num_data_records(
                signal_durations[0],
                self.data_record_duration,
            )
            signal_lengths = [len(s._digital) for s in signals]
            if any(l % num_data_records for l in signal_lengths):
                raise ValueError(
                    f"Not all signal lengths can be split into {num_data_records} data records: {signal_lengths}"
                )
        self._num_data_records = Edf.num_data_records.encode(num_data_records)

    def _parse_signal_headers(self, raw_signal_headers: bytes) -> tuple[EdfSignal, ...]:
        raw_headers_split: dict[str, list[bytes]] = {}
        start = 0
        for header_name, length in get_header_fields(EdfSignal):
            end = start + length * self.num_signals
            raw_header = raw_signal_headers[start:end]
            raw_headers_split[header_name] = [
                raw_header[i : length + i] for i in range(0, len(raw_header), length)
            ]
            start = end
        signals = []
        for i in range(self.num_signals):
            raw_signal_header = {
                key: raw_headers_split[key][i] for key in raw_headers_split
            }
            try:
                sampling_frequency = (
                    int(raw_signal_header["samples_per_data_record"])
                    / self.data_record_duration
                )
            except ZeroDivisionError:
                if raw_signal_header["label"].rstrip() == b"EDF Annotations":
                    sampling_frequency = 0
            signals.append(
                EdfSignal._from_raw_header(sampling_frequency, **raw_signal_header)
            )
        return tuple(signals)

    def write(self, target: Path | str | io.BufferedWriter | io.BytesIO) -> None:
        """
        Write an Edf to a file or file-like object.

        Parameters
        ----------
        target : Path | str | io.BufferedWriter | io.BytesIO
            The file location (path object or string) or file-like object to write to.
        """
        if self.num_data_records == -1:
            warnings.warn("num_data_records=-1, determining correct value from data")
            num_data_records = _calculate_num_data_records(
                len(self.signals[0]._digital) * self.signals[0].sampling_frequency,
                self.data_record_duration,
            )
        else:
            num_data_records = self.num_data_records
        for signal in self.signals:
            signal._samples_per_data_record = EdfSignal.samples_per_data_record.encode(  # type: ignore[attr-defined]
                len(signal._digital) // num_data_records
            )
        header_records = []
        for header_name, _ in get_header_fields(Edf):
            header_records.append(getattr(self, "_" + header_name))
        for header_name, _ in get_header_fields(EdfSignal):
            for signal in self.signals:
                header_records.append(getattr(signal, "_" + header_name))
        header_record = b"".join(header_records)

        lens = [signal.samples_per_data_record for signal in self.signals]
        ends = np.cumsum(lens)
        starts = ends - lens
        data_record = np.empty((num_data_records, sum(lens)), dtype=np.int16)
        for signal, start, end in zip(self.signals, starts, ends):
            data_record[:, start:end] = signal._digital.reshape((-1, end - start))

        if isinstance(target, str):
            target = Path(target)
        if isinstance(target, io.BufferedWriter):
            target.write(header_record)
            data_record.tofile(target)
        elif isinstance(target, io.BytesIO):
            target.write(header_record)
            target.write(data_record.tobytes())
        else:
            with target.expanduser().open("wb") as file:
                file.write(header_record)
                data_record.tofile(file)

    @property
    def labels(self) -> tuple[str, ...]:
        """
        The labels of all signals contained in the Edf.

        Returns
        -------
        tuple[str, ...]
            The labels, in order of the signals.
        """
        return tuple(s.label for s in self.signals)

    def get_signal(self, label: str) -> EdfSignal:
        """
        Retrieve a single signal by its label.

        The label has to be unique - a ValueError is raised if it is ambiguous or does
        not exist.

        Parameters
        ----------
        label : str
            A label identifying a single signal

        Returns
        -------
        EdfSignal
            The signal corresponding to the given label.
        """
        count = self.labels.count(label)
        if count == 0:
            raise ValueError(
                f"No signal with label {label!r}, possible options: {self.labels}"
            )
        if count > 1:
            indices = [i for i, l in enumerate(self.labels) if l == label]
            raise ValueError(f"Ambiguous label {label!r} identifies indices {indices}")
        return self.signals[self.labels.index(label)]

    @property
    def patient(self) -> Patient:
        """
        Parsed object representation of the local patient identification.

        See :class:`Patient` for information on its attributes.
        """
        return Patient._from_str(self.local_patient_identification)

    @patient.setter
    def patient(self, patient: Patient) -> None:
        self.local_patient_identification = patient._to_str()

    @property
    def recording(self) -> Recording:
        """
        Parsed object representation of the local recording identification.

        See :class:`Recording` for information on its attributes.
        """
        return Recording._from_str(self.local_recording_identification)

    @recording.setter
    def recording(self, recording: Recording) -> None:
        self._set_startdate_with_recording(recording)
        self.local_recording_identification = recording._to_str()

    @property
    def startdate(self) -> datetime.date:
        """
        Recording startdate.

        If the :attr:`local_recording_identification` conforms to the EDF+ standard, the
        startdate provided there is used. If not, this falls back to the legacy
        :attr:`startdate` field. If both differ, a warning is issued and the EDF+ field
        is preferred. Raises an `AnonymizedDateError` if the EDF+ field is anonymized
        (i.e., begins with `Startdate X`).
        """
        with contextlib.suppress(Exception):
            if self._startdate != self.recording.startdate:
                warnings.warn(
                    f"Different values in startdate fields: {self._startdate}, {self.recording.startdate}"
                )
        try:
            return self.recording.startdate
        except AnonymizedDateError:
            raise
        except ValueError:
            return self._startdate

    @startdate.setter
    def startdate(self, startdate: datetime.date) -> None:
        self._startdate = startdate
        try:
            self.recording.startdate  # noqa: B018
        except AnonymizedDateError:
            pass
        except Exception:
            return
        recording_subfields = self.local_recording_identification.split()
        recording_subfields[1] = encode_edfplus_date(startdate)
        self.local_recording_identification = " ".join(recording_subfields)

    @property
    def _subsecond_offset(self) -> float:
        try:
            timekeeping_raw = self._timekeeping_signal._digital.tobytes()
            first_data_record = timekeeping_raw[: timekeeping_raw.find(b"\x00") + 1]
            return _EdfAnnotationsDataRecord.from_bytes(first_data_record).tals[0].onset
        except StopIteration:
            return 0

    @property
    def starttime(self) -> datetime.time:
        """
        Recording starttime.

        In EDF+ files, microsecond accuracy is supported.
        """
        subsecond_offset = self._subsecond_offset
        try:
            return self._starttime.replace(
                microsecond=round(subsecond_offset * 1000000)
            )
        except ValueError as e:
            raise ValueError(
                f"Subsecond offset in first annotation must be 0.X, is {subsecond_offset}"
            ) from e

    @starttime.setter
    def starttime(self, starttime: datetime.time) -> None:
        onset_change = starttime.microsecond / 1000000 - self._subsecond_offset
        self._starttime = starttime.replace(microsecond=0)
        if starttime.microsecond != self.starttime.microsecond:
            timekeeping_signal = self._timekeeping_signal
            data_records = []
            for data_record in timekeeping_signal._digital.reshape(
                (-1, timekeeping_signal.samples_per_data_record)
            ):
                annot_dr = _EdfAnnotationsDataRecord.from_bytes(data_record.tobytes())
                for tal in annot_dr.tals:
                    tal.onset = round(tal.onset + onset_change, 12)
                data_records.append(annot_dr.to_bytes())
            maxlen = max(len(data_record) for data_record in data_records)
            if maxlen % 2:
                maxlen += 1
            raw = b"".join(dr.ljust(maxlen, b"\x00") for dr in data_records)
            timekeeping_signal._samples_per_data_record = EdfSignal.samples_per_data_record.encode(  # type: ignore[attr-defined]
                maxlen // 2
            )
            timekeeping_signal._sampling_frequency = (
                maxlen // 2 * self.data_record_duration
            )
            timekeeping_signal._digital = np.frombuffer(raw, dtype=np.int16)

    def _set_startdate_with_recording(self, recording: Recording) -> None:
        try:
            self._startdate = recording.startdate
        except AnonymizedDateError:
            self._startdate = datetime.date(1985, 1, 1)

    @property
    def data_record_duration(self) -> float:
        """Duration of each data record in seconds."""
        return self._data_record_duration

    def update_data_record_duration(
        self,
        data_record_duration: float,
        method: Literal["strict", "pad", "truncate"] = "strict",
    ) -> None:
        """
        Update the data record duration.

        This operation will fail if the new duration is incompatible with the current
        sampling frequencies.

        Parameters
        ----------
        data_record_duration : float
            The new data record duration in seconds.
        method : `{"strict", "pad", "truncate"}`, default: `"strict"`
            How to handle the case where the new duration does not divide the Edf
            duration evenly

            - "strict": Raise a ValueError
            - "pad": Pad the data with zeros to the next compatible duration. If zero
              is outside the physical range, data is padded with the physical minimum.
            - "truncate": Truncate the data to the previous compatible duration (might
              lead to loss of data)
        """
        if data_record_duration == self.data_record_duration:
            return
        if data_record_duration <= 0:
            raise ValueError(
                f"Data record duration must be positive, got {data_record_duration}"
            )
        if not any(
            signal for signal in self.signals if signal not in self._annotation_signals
        ):
            raise ValueError(
                "Data record duration must be zero for annotation-only files"
            )
        for signal in self.signals:
            if signal in self._annotation_signals:
                continue
            spr = signal.sampling_frequency * data_record_duration
            if spr % 1:
                raise ValueError(
                    f"Cannot set data record duration to {data_record_duration}: Incompatible sampling frequency {signal.sampling_frequency} Hz"
                )

        num_data_records = self._pad_or_truncate_signals(data_record_duration, method)
        self._update_record_duration_in_annotation_signals(
            data_record_duration, num_data_records
        )
        self._data_record_duration = data_record_duration
        self._num_data_records = Edf.num_data_records.encode(num_data_records)

    def _pad_or_truncate_signals(
        self, data_record_duration: float, method: Literal["strict", "pad", "truncate"]
    ) -> int:
        if method == "pad":
            new_duration = (
                ceil(self.duration / data_record_duration) * data_record_duration
            )
            self._pad_or_truncate_data(new_duration)
            return round(new_duration / data_record_duration)
        if method == "truncate":
            new_duration = (
                floor(self.duration / data_record_duration) * data_record_duration
            )
            self._pad_or_truncate_data(new_duration)
            return round(new_duration / data_record_duration)
        return _calculate_num_data_records(self.duration, data_record_duration)

    def _update_record_duration_in_annotation_signals(
        self, data_record_duration: float, num_data_records: int
    ) -> None:
        signals = list(self.signals)
        for idx, signal in enumerate(self._signals):
            if signal not in self._annotation_signals:
                continue
            annotations = []
            for data_record in signal._digital.reshape(
                (-1, signal.samples_per_data_record)
            ):
                annot_dr = _EdfAnnotationsDataRecord.from_bytes(data_record.tobytes())
                if signal is self._timekeeping_signal:
                    annotations.extend(annot_dr.annotations[1:])
                else:
                    annotations.extend(annot_dr.annotations)
            signals[idx] = _create_annotations_signal(
                [
                    EdfAnnotation(a.onset - self._subsecond_offset, a.duration, a.text)
                    for a in annotations
                ],
                num_data_records=num_data_records,
                data_record_duration=data_record_duration,
                with_timestamps=signal is self._timekeeping_signal,
                subsecond_offset=self._subsecond_offset,
            )
        self._signals = tuple(signals)

    def _pad_or_truncate_data(self, new_duration: float) -> None:
        for signal in self.signals:
            if signal in self._annotation_signals:
                continue
            n_samples = round(new_duration * signal.sampling_frequency)
            diff = n_samples - len(signal._digital)
            if diff > 0:
                physical_pad_value = 0.0
                if signal.physical_min > 0 or signal.physical_max < 0:
                    physical_pad_value = signal.physical_min
                signal._set_data(
                    np.pad(signal.data, (0, diff), constant_values=physical_pad_value)
                )
            elif diff < 0:
                signal._set_data(signal.data[:diff])

    def anonymize(self) -> None:
        """
        Anonymize a recording.

        Header fields are modified as follows:
          - local patient identification is set to `X X X X`
          - local recording identification is set to `Startdate X X X X`
          - startdate is set to `01.01.85`
          - starttime is set to `00.00.00`

        For EDF+ files, subsecond starttimes specified via an annotations signal are
        removed.
        """
        self.patient = Patient()
        self.recording = Recording()
        self.starttime = datetime.time(0, 0, 0)

    def drop_signals(self, drop: Iterable[int | str]) -> None:
        """
        Drop signals by index or label.

        Signal indices (int) and labels (str) can be provided in the same iterable. For
        ambiguous labels, all corresponding signals are dropped. Raises a ValueError if
        at least one of the provided identifiers does not correspond to a signal.

        Parameters
        ----------
        drop : Iterable[int  |  str]
            The signals to drop, identified by index or label.
        """
        if isinstance(drop, str):
            drop = [drop]
        selected: list[EdfSignal] = []
        dropped: list[int | str] = []
        try:
            timekeeping_signal = self._timekeeping_signal
        except StopIteration:
            timekeeping_signal = None
        for i, signal in enumerate(self.signals):
            if i in drop or signal.label in drop:
                if signal is timekeeping_signal:
                    raise ValueError("Can not drop EDF+ timekeeping signal.")
                dropped.append(i)
                dropped.append(signal.label)
            else:
                selected.append(signal)
        if not_dropped := set(drop) - set(dropped):
            raise ValueError(f"No signal found with index/label {not_dropped}")
        self._signals = tuple(selected)
        self._bytes_in_header_record = Edf.bytes_in_header_record.encode(
            256 * (len(selected) + 1)
        )
        self._num_signals = Edf.num_signals.encode(len(selected))

    def append_signals(self, new_signals: EdfSignal | Iterable[EdfSignal]) -> None:
        """
        Append one or more signal(s) to the Edf recording, after all existing ones.

        Every signal must be compatible with the current `data_record_duration` and all
        signal durations must match the overall recording duration.

        Parameters
        ----------
        new_signals : EdfSignal | Iterable[EdfSignal]
            The signal(s) to add.
        """
        if isinstance(new_signals, EdfSignal):
            new_signals = [new_signals]
        self._set_signals((*self.signals, *new_signals))

    @property
    def _annotation_signals(self) -> Iterable[EdfSignal]:
        return (signal for signal in self.signals if signal.label == "EDF Annotations")

    @property
    def _timekeeping_signal(self) -> EdfSignal:
        return next(iter(self._annotation_signals))

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        return self.num_data_records * self.data_record_duration

    @property
    def annotations(self) -> tuple[EdfAnnotation, ...]:
        """
        All annotations contained in the Edf, sorted chronologically.

        Does not include timekeeping annotations.
        """
        annotations: list[EdfAnnotation] = []
        for i, signal in enumerate(self._annotation_signals):
            for data_record in signal._digital.reshape(
                (-1, signal.samples_per_data_record)
            ):
                annot_dr = _EdfAnnotationsDataRecord.from_bytes(data_record.tobytes())
                if i == 0:
                    # from https://www.edfplus.info/specs/edfplus.html#timekeeping:
                    # The first annotation of the first 'EDF Annotations' signal in each
                    # data record is empty, but its timestamp specifies how many seconds
                    # after the file startdate/time that data record starts.
                    annotations.extend(annot_dr.annotations[1:])
                else:
                    annotations.extend(annot_dr.annotations)
        subsecond_offset = self._subsecond_offset
        annotations = [
            EdfAnnotation(
                round(ann.onset - subsecond_offset, 12), ann.duration, ann.text
            )
            for ann in annotations
        ]
        return tuple(sorted(annotations))

    def drop_annotations(self, text: str) -> None:
        """
        Drop annotations with a given text.

        Parameters
        ----------
        text : str
            All annotations whose text exactly matches this parameter are removed.
        """
        for signal in self._annotation_signals:
            for data_record in signal._digital.reshape(
                (-1, signal.samples_per_data_record)
            ):
                annotations = _EdfAnnotationsDataRecord.from_bytes(
                    data_record.tobytes()
                )
                annotations.drop_annotations_with_text(text)
                data_record[:] = np.frombuffer(
                    annotations.to_bytes().ljust(len(data_record) * 2, b"\x00"),
                    dtype=np.int16,
                )

    def to_bytes(self) -> bytes:
        """
        Convert an Edf to a `bytes` object.

        Returns
        -------
        bytes
            The binary representation of the Edf object (i.e., what a file created with
            `Edf.write` would contain).
        """
        stream = io.BytesIO()
        self.write(stream)
        stream.seek(0)
        return stream.read()

    def slice_between_seconds(
        self,
        start: float,
        stop: float,
        *,
        keep_all_annotations: bool = False,
    ) -> None:
        """
        Slice to the interval between two times.

        The sample point corresponding to `stop` is excluded. `start` and `stop` are
        given in seconds from recording start and have to correspond exactly to a sample
        time in all non-annotation signals.

        Parameters
        ----------
        start : float
            Start time in seconds from recording start.
        stop : float
            Stop time in seconds from recording start.
        keep_all_annotations : bool, default: False
            If set to `True`, annotations outside the selected time interval are kept.
        """
        signals: list[EdfSignal] = []
        self._verify_seconds_inside_recording_time(start)
        self._verify_seconds_inside_recording_time(stop)
        self._verify_seconds_coincide_with_sample_time(start)
        self._verify_seconds_coincide_with_sample_time(stop)
        self._num_data_records = Edf.num_data_records.encode(
            int((stop - start) / self.data_record_duration)
        )
        for signal in self.signals:
            if signal.label == "EDF Annotations":
                signals.append(
                    self._slice_annotations_signal(
                        signal,
                        start=start,
                        stop=stop,
                        keep_all_annotations=keep_all_annotations,
                    )
                )
            else:
                start_index = start * signal.sampling_frequency
                stop_index = stop * signal.sampling_frequency
                signal._digital = signal._digital[int(start_index) : int(stop_index)]
                signals.append(signal)
        self._set_signals(signals)
        self._shift_startdatetime(int(start))

    def slice_between_annotations(
        self,
        start_text: str,
        stop_text: str,
        *,
        keep_all_annotations: bool = False,
    ) -> None:
        """
        Slice to the interval between two EDF+ annotations.

        The sample point corresponding to the onset of the annotation identified by
        `stop_text` is excluded. `start_text` and `stop_text` each have to uniquely
        identify a single annotation, whose onset corresponds exactly to a sample time
        in all non-annotation signals.

        Parameters
        ----------
        start_text : str
            Text identifying the start annotation.
        stop_text : str
            Text identifying the stop annotation.
        keep_all_annotations : bool, default: False
            If set to `True`, annotations outside the selected time interval are kept.
        """
        self.slice_between_seconds(
            self._get_annotation_by_text(start_text).onset,
            self._get_annotation_by_text(stop_text).onset,
            keep_all_annotations=keep_all_annotations,
        )

    def _get_annotation_by_text(self, text: str) -> EdfAnnotation:
        matches = []
        for annotation in self.annotations:
            if annotation.text == text:
                matches.append(annotation)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous annotation text {text!r}, found {len(matches)} matches"
            )
        raise ValueError(f"No annotation found with text {text!r}")

    def _verify_seconds_inside_recording_time(self, seconds: float) -> None:
        if not 0 <= seconds <= self.duration:
            raise ValueError(
                f"{seconds} is an invalid slice time for recording duration {self.duration}"
            )

    def _verify_seconds_coincide_with_sample_time(self, seconds: float) -> None:
        for i, signal in enumerate(self.signals):
            if signal.label == "EDF Annotations":
                continue
            index = seconds * signal.sampling_frequency
            if index != int(index):
                raise ValueError(
                    f"{seconds}s is not a sample time of signal {i} ({signal.label}) with fs={signal.sampling_frequency}Hz"
                )

    def _shift_startdatetime(self, seconds: float) -> None:
        timedelta = datetime.timedelta(seconds=seconds)
        try:
            startdate = self.startdate
            startdate_anonymized = False
        except AnonymizedDateError:
            startdate = datetime.date.fromtimestamp(0)
            startdate_anonymized = True
        startdatetime = datetime.datetime.combine(startdate, self.starttime)
        startdatetime += timedelta
        if not startdate_anonymized:
            self.startdate = startdatetime.date()
        self.starttime = startdatetime.time()

    def copy(self) -> Edf:
        """
        Create a deep copy of the Edf.

        Returns
        -------
        Edf
            The copied Edf object.
        """
        return copy.deepcopy(self)

    def _slice_annotations_signal(
        self,
        signal: EdfSignal,
        *,
        start: float,
        stop: float,
        keep_all_annotations: bool,
    ) -> EdfSignal:
        is_timekeeping_signal = signal == self._timekeeping_signal
        annotations: list[EdfAnnotation] = []
        for data_record in signal._digital.reshape(
            (-1, signal.samples_per_data_record)
        ):
            annot_dr = _EdfAnnotationsDataRecord.from_bytes(data_record.tobytes())
            if is_timekeeping_signal:
                annotations.extend(annot_dr.annotations[1:])
            else:
                annotations.extend(annot_dr.annotations)
        annotations = [
            EdfAnnotation(round(a.onset - start, 12), a.duration, a.text)
            for a in annotations
            if keep_all_annotations or start <= a.onset < stop
        ]
        return _create_annotations_signal(
            annotations,
            num_data_records=self.num_data_records,
            data_record_duration=self.data_record_duration,
            with_timestamps=is_timekeeping_signal,
            subsecond_offset=self._subsecond_offset + start - int(start),
        )


def _create_annotations_signal(
    annotations: Iterable[EdfAnnotation],
    *,
    num_data_records: int,
    data_record_duration: float,
    with_timestamps: bool = True,
    subsecond_offset: float = 0,
) -> EdfSignal:
    data_record_starts = np.arange(num_data_records) * data_record_duration
    annotations = sorted(annotations)
    data_records = []
    for i, start in enumerate(data_record_starts):
        end = start + data_record_duration
        tals: list[_EdfTAL] = []
        if with_timestamps:
            tals.append(_EdfTAL(np.round(start + subsecond_offset, 12), None, [""]))
        for ann in annotations:
            if (
                (i == 0 and ann.onset < 0)
                or (i == (num_data_records - 1) and end <= ann.onset)
                or (start <= ann.onset < end)
            ):
                tals.append(
                    _EdfTAL(
                        np.round(ann.onset + subsecond_offset, 12),
                        ann.duration,
                        [ann.text],
                    )
                )
        data_records.append(_EdfAnnotationsDataRecord(tals).to_bytes())
    maxlen = max(len(data_record) for data_record in data_records)
    if maxlen % 2:
        maxlen += 1
    raw = b"".join(dr.ljust(maxlen, b"\x00") for dr in data_records)
    divisor = data_record_duration if data_record_duration else 1
    signal = EdfSignal(
        np.arange(1.0),  # placeholder signal, as argument `data` is non-optional
        sampling_frequency=maxlen // 2 / divisor,
        label="EDF Annotations",
        physical_range=(-32768, 32767),
    )
    signal._samples_per_data_record = EdfSignal.samples_per_data_record.encode(  # type: ignore[attr-defined]
        maxlen // 2
    )
    signal._digital = np.frombuffer(raw, dtype=np.int16).copy()
    return signal


@dataclass
class _EdfTAL:
    onset: float
    duration: float | None
    texts: list[str]

    def to_bytes(self) -> bytes:
        timing = encode_annotation_onset(self.onset)
        if self.duration is not None:
            timing += f"\x15{encode_annotation_duration(self.duration)}"
        texts_joined = "\x14".join(self.texts)
        return f"{timing}\x14{texts_joined}\x14".encode()


@dataclass
class _EdfAnnotationsDataRecord:
    tals: list[_EdfTAL]

    def to_bytes(self) -> bytes:
        return b"\x00".join(tal.to_bytes() for tal in self.tals) + b"\x00"

    @classmethod
    def from_bytes(cls, raw: bytes) -> _EdfAnnotationsDataRecord:
        tals: list[_EdfTAL] = []
        matches: list[tuple[str, str, str]] = _ANNOTATIONS_PATTERN.findall(raw.decode())
        if not matches and raw.replace(b"\x00", b""):
            raise ValueError(f"No valid annotations found in {raw!r}")
        for onset, duration, texts in matches:
            tals.append(
                _EdfTAL(
                    float(onset),
                    float(duration) if duration else None,
                    list(texts.split("\x14")),
                )
            )
        return cls(tals)

    @property
    def annotations(self) -> list[EdfAnnotation]:
        return [
            EdfAnnotation(tal.onset, tal.duration, text)
            for tal in self.tals
            for text in tal.texts
        ]

    def drop_annotations_with_text(self, text: str) -> None:
        for tal in self.tals:
            while text in tal.texts:
                tal.texts.remove(text)
        self.tals = [tal for tal in self.tals if tal.texts]


def _calculate_num_data_records(
    signal_duration: float,
    data_record_duration: float,
) -> int:
    if data_record_duration < 0:
        raise ValueError(
            f"data_record_duration must be positive, got {data_record_duration}"
        )
    for f in (lambda x: x, lambda x: Decimal(str(x))):
        required_num_data_records = f(signal_duration) / f(data_record_duration)
        if required_num_data_records == int(required_num_data_records):
            return int(required_num_data_records)
    raise ValueError(
        f"Signal duration of {signal_duration}s is not exactly divisible by data_record_duration of {data_record_duration}s"
    )


def _calculate_data_record_duration(signals: Sequence[EdfSignal]) -> float:
    fs = (Fraction(s.sampling_frequency).limit_denominator(99999999) for s in signals)
    return math.lcm(*(fs_.denominator for fs_ in fs))


@singledispatch
def _read_edf(edf_file: Any) -> Edf:
    raise NotImplementedError(f"Can not read EDF from {type(edf_file)}")


@_read_edf.register
def _(edf_file: Path) -> Edf:
    edf = object.__new__(Edf)
    edf_file = edf_file.expanduser().resolve()
    with edf_file.open("rb") as file:
        edf._read_header(file)
    edf._load_data(edf_file)
    return edf


@_read_edf.register
def _(edf_file: str) -> Edf:
    return _read_edf(Path(edf_file))


@_read_edf.register
def _(edf_file: bytes) -> Edf:
    return _read_edf(io.BytesIO(edf_file))


# explicit register is necessary because type unions are only supported from Python 3.11
@_read_edf.register(io.BufferedReader)
@_read_edf.register(io.BufferedRandom)
@_read_edf.register(io.BytesIO)
def _(edf_file: io.BufferedReader | io.BufferedRandom | io.BytesIO) -> Edf:
    edf = object.__new__(Edf)
    edf._read_header(edf_file)
    edf._load_data(edf_file)
    return edf


# Pyright loses information about parameters for singledispatch functions. Hiding it
# behind this normal function makes things work again.
def read_edf(edf_file: Path | str | io.BufferedReader | io.BytesIO | bytes) -> Edf:
    """
    Read an EDF file into an :class:`Edf` object.

    Parameters
    ----------
    edf_file : Path | str | io.BufferedReader | io.BytesIO
        The file location (path object or string) or file-like object to read from.

    Returns
    -------
    Edf
        The resulting :class:`Edf` object.
    """
    return _read_edf(edf_file)

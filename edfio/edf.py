from __future__ import annotations

import contextlib
import copy
import datetime
import io
import math
import tempfile
import warnings
from collections.abc import Iterable, Sequence
from decimal import Decimal
from fractions import Fraction
from functools import singledispatch
from math import ceil, floor
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from edfio._header_field import (
    decode_date,
    decode_float,
    decode_str,
    decode_time,
    encode_date,
    encode_float,
    encode_int,
    encode_str,
    encode_time,
)
from edfio._lazy_loading import LazyLoader
from edfio.edf_annotations import (
    EdfAnnotation,
    _create_annotations_signal,
    _EdfAnnotationsDataRecord,
)
from edfio.edf_header import (
    AnonymizedDateError,
    Patient,
    Recording,
    _encode_edfplus_date,
)
from edfio.edf_signal import EdfSignal


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

    _header_fields = (
        ("version", 8),
        ("local_patient_identification", 80),
        ("local_recording_identification", 80),
        ("startdate", 8),
        ("starttime", 8),
        ("bytes_in_header_record", 8),
        ("reserved", 44),
        ("num_data_records", 8),
        ("data_record_duration", 8),
        ("num_signals", 4),
    )
    _signals: tuple[EdfSignal, ...]
    _cached_onset_times: NDArray[np.float64] | None

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

        self._set_data_record_duration(data_record_duration)
        self._set_num_data_records_with_signals(signals)
        self._set_version(0)
        self.local_patient_identification = patient._to_str()
        self.local_recording_identification = recording._to_str()
        self._set_startdate_with_recording(recording)
        self._starttime = encode_time(starttime.replace(microsecond=0))
        self._set_reserved("")
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
            self._set_reserved("EDF+C")
        self._set_signals(signals)
        self._header_encoding = "ascii"
        self._cached_onset_times = None

    def __repr__(self) -> str:
        signals_text = f"{len(self.signals)} signal"
        if len(self.signals) != 1:
            signals_text += "s"
        annotations_text = f"{len(self.annotations)} annotation"
        if len(self.annotations) != 1:
            annotations_text += "s"
        return f"<Edf {signals_text} {annotations_text}>"

    def _set_version(self, version: int) -> None:
        self._version = encode_int(version, 8)

    def _set_bytes_in_header_record(self, bytes_in_header_record: int) -> None:
        self._bytes_in_header_record = encode_int(bytes_in_header_record, 8)

    def _set_reserved(self, reserved: str) -> None:
        self._reserved = encode_str(reserved, 44)

    def _set_num_data_records(self, num_data_records: int) -> None:
        self._num_data_records = encode_int(num_data_records, 8)
        self._invalidate_onset_cache()

    def _set_data_record_duration(self, data_record_duration: float) -> None:
        self._data_record_duration = encode_float(data_record_duration)
        self._invalidate_onset_cache()

    def _set_num_signals(self, num_signals: int) -> None:
        self._num_signals = encode_int(num_signals, 4)

    @property
    def version(self) -> int:
        """EDF version, always `0`."""
        return int(decode_str(self._version))

    @property
    def local_patient_identification(self) -> str:
        """
        Unparsed string representation of the legacy local patient identification.

        See Also
        --------
        patient: Parsed representation, as a :class:`Patient` object.
        """
        return decode_str(self._local_patient_identification, self._header_encoding)

    @local_patient_identification.setter
    def local_patient_identification(self, value: str) -> None:
        self._local_patient_identification = encode_str(value, 80)

    @property
    def local_recording_identification(self) -> str:
        """
        Unparsed string representation of the legacy local recording identification.

        See Also
        --------
        recording: Parsed representation, as a :class:`Recording` object.
        """
        return decode_str(self._local_recording_identification, self._header_encoding)

    @local_recording_identification.setter
    def local_recording_identification(self, value: str) -> None:
        self._local_recording_identification = encode_str(value, 80)

    def _load_data(
        self, file: Path | io.BufferedReader | io.BytesIO, *, lazy_load_data: bool
    ) -> None:
        if lazy_load_data and not isinstance(file, Path):
            raise ValueError("Lazy loading is only supported for local file paths")
        lens = [signal.samples_per_data_record for signal in self._signals]
        datarecord_len = sum(lens)
        truncated = False
        if not isinstance(file, Path):
            data_bytes = file.read()
            actual_records = len(data_bytes) // (datarecord_len * 2)
            if actual_records * datarecord_len * 2 < len(data_bytes):
                truncated = True
            datarecords = np.frombuffer(
                data_bytes, dtype=np.int16, count=actual_records * datarecord_len
            )
            datarecords.shape = (actual_records, datarecord_len)
        else:
            remaining_bytes = file.stat().st_size - self.bytes_in_header_record
            actual_records = remaining_bytes // (datarecord_len * 2)
            if actual_records * datarecord_len * 2 < remaining_bytes:
                truncated = True
            datarecords = np.memmap(
                file,
                dtype=np.int16,
                mode="r",
                offset=self.bytes_in_header_record,
                shape=(actual_records, datarecord_len),
            )
        if truncated:
            warnings.warn(
                "Incomplete data record at the end of the EDF file. Data was truncated."
            )
        if self.num_data_records != actual_records:
            warnings.warn(
                f"EDF header indicates {self.num_data_records} data records, but file contains {actual_records} records. Updating header."
            )
            self._set_num_data_records(actual_records)
        ends = np.cumsum(lens)
        starts = ends - lens

        for signal, start, end in zip(self._signals, starts, ends):
            if lazy_load_data:
                signal._lazy_loader = LazyLoader(datarecords, start, end)
            else:
                signal._digital = datarecords[:, start:end].flatten()

    def _read_header(
        self,
        buffer: io.BufferedReader | io.BytesIO,
        header_encoding: str,
    ) -> None:
        self._header_encoding = header_encoding
        self._cached_onset_times = None
        for header_name, length in Edf._header_fields:
            setattr(self, "_" + header_name, buffer.read(length))
        self._signals = self._parse_signal_headers(
            buffer.read(256 * self._total_num_signals), header_encoding
        )

    @property
    def signals(self) -> tuple[EdfSignal, ...]:
        """
        Ordinary signals contained in the recording.

        Annotation signals are excluded. Individual signals can not be removed, added,
        or replaced by modifying this property. Use :meth:`Edf.append_signals`,
        :meth:`Edf.drop_signals`, or :attr:`EdfSignal.data`, respectively.
        """
        return tuple(s for s in self._signals if s.label != "EDF Annotations")

    def _set_signals(self, signals: Sequence[EdfSignal]) -> None:
        signals = tuple(signals)
        self._set_num_data_records_with_signals(signals)
        self._signals = signals
        self._set_bytes_in_header_record(256 * (len(signals) + 1))
        self._set_num_signals(len(signals))
        if all(s.label == "EDF Annotations" for s in signals):
            self._set_data_record_duration(0)
        self._invalidate_onset_cache()

    def _set_num_data_records_with_signals(
        self,
        signals: Sequence[EdfSignal],
    ) -> None:
        if not signals:
            num_data_records = 1
        else:
            signal_durations = [
                round(len(s.digital) / s.sampling_frequency, 12) for s in signals
            ]
            if any(v != signal_durations[0] for v in signal_durations[1:]):
                raise ValueError(
                    f"Inconsistent signal durations (in seconds): {signal_durations}"
                )
            num_data_records = _calculate_num_data_records(
                signal_durations[0],
                self.data_record_duration,
            )
            signal_lengths = [len(s.digital) for s in signals]
            if any(l % num_data_records for l in signal_lengths):
                raise ValueError(
                    f"Not all signal lengths can be split into {num_data_records} data records: {signal_lengths}"
                )
        self._set_num_data_records(num_data_records)

    def _parse_signal_headers(
        self,
        raw_signal_headers: bytes,
        header_encoding: str,
    ) -> tuple[EdfSignal, ...]:
        raw_headers_split: dict[str, list[bytes]] = {}
        start = 0
        for header_name, length in EdfSignal._header_fields:
            end = start + length * self._total_num_signals
            raw_header = raw_signal_headers[start:end]
            raw_headers_split[header_name] = [
                raw_header[i : length + i] for i in range(0, len(raw_header), length)
            ]
            start = end
        signals = []
        for i in range(self._total_num_signals):
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
                else:
                    raise
            signals.append(
                EdfSignal._from_raw_header(
                    sampling_frequency,
                    **raw_signal_header,
                    header_encoding=header_encoding,
                )
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
                len(self._signals[0].digital) * self._signals[0].sampling_frequency,
                self.data_record_duration,
            )
        else:
            num_data_records = self.num_data_records
        for signal in self._signals:
            signal._set_samples_per_data_record(len(signal.digital) // num_data_records)
        header_records = []
        for header_name, _ in Edf._header_fields:
            header_records.append(getattr(self, "_" + header_name))
        for header_name, _ in EdfSignal._header_fields:
            for signal in self._signals:
                header_records.append(getattr(signal, "_" + header_name))
        header_record = b"".join(header_records)

        lens = [signal.samples_per_data_record for signal in self._signals]
        ends = np.cumsum(lens)
        starts = ends - lens
        data_record = np.empty((num_data_records, sum(lens)), dtype=np.int16)
        for signal, start, end in zip(self._signals, starts, ends):
            data_record[:, start:end] = signal.digital.reshape((-1, end - start))

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
        # account for dates outside of 1985 - 2084 with .yy listed for the two digit year
        if b"yy" in self._startdate:
            legacy_startdate = self.recording.startdate
        else:
            legacy_startdate = decode_date(self._startdate)

        with contextlib.suppress(Exception):
            if legacy_startdate != self.recording.startdate:
                warnings.warn(
                    f"Different values in startdate fields: {legacy_startdate}, {self.recording.startdate}"
                )
        try:
            return self.recording.startdate
        except AnonymizedDateError:
            raise
        except ValueError:
            return legacy_startdate

    @startdate.setter
    def startdate(self, startdate: datetime.date) -> None:
        self._startdate = encode_date(startdate)
        try:
            self.recording.startdate  # noqa: B018
        except AnonymizedDateError:
            pass
        except Exception:
            return
        recording_subfields = self.local_recording_identification.split()
        recording_subfields[1] = _encode_edfplus_date(startdate)
        self.local_recording_identification = " ".join(recording_subfields)

    @property
    def _subsecond_offset(self) -> float:
        try:
            timekeeping_raw = self._timekeeping_signal.digital.tobytes()
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
            return decode_time(self._starttime).replace(
                microsecond=round(subsecond_offset * 1000000)
            )
        except ValueError as e:
            raise ValueError(
                f"Subsecond offset in first annotation must be 0.X, is {subsecond_offset}"
            ) from e

    @starttime.setter
    def starttime(self, starttime: datetime.time) -> None:
        onset_change = starttime.microsecond / 1000000 - self._subsecond_offset
        self._starttime = encode_time(starttime.replace(microsecond=0))
        if starttime.microsecond != self.starttime.microsecond:
            timekeeping_signal = self._timekeeping_signal
            data_records = []
            for data_record in timekeeping_signal.digital.reshape(
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
            timekeeping_signal._set_samples_per_data_record(maxlen // 2)
            timekeeping_signal._sampling_frequency = (
                maxlen // 2 * self.data_record_duration
            )
            timekeeping_signal._digital = np.frombuffer(raw, dtype=np.int16)

    def _set_startdate_with_recording(self, recording: Recording) -> None:
        try:
            startdate = recording.startdate
        except AnonymizedDateError:
            startdate = datetime.date(1985, 1, 1)
        self._startdate = encode_date(startdate)

    @property
    def bytes_in_header_record(self) -> int:
        """Number of bytes in the header record."""
        return int(decode_str(self._bytes_in_header_record))

    @property
    def reserved(self) -> str:
        """Decodes to str Reserved header field for EDF files.

        Used in EDF+ files: first 5 bytes of this
        field to contain 'EDF+C' or 'EDF+D' for EDF+ files, else unused
        empty spaces.
        """
        return decode_str(self._reserved)

    @property
    def edf_format(self) -> Literal["EDF", "EDF+C", "EDF+D"]:
        """The EDF format variant: 'EDF', 'EDF+C', or 'EDF+D'."""
        reserved_start = decode_str(self._reserved[:5]).strip()
        if reserved_start == "EDF+C":
            return "EDF+C"
        if reserved_start == "EDF+D":
            return "EDF+D"
        return "EDF"

    @property
    def is_edf_plus(self) -> bool:
        """True if this is an EDF+ file (EDF+C or EDF+D)."""
        return self.edf_format in ("EDF+C", "EDF+D")

    @property
    def is_discontinuous(self) -> bool:
        """True if this is a discontinuous EDF+D file."""
        return self.edf_format == "EDF+D"

    @property
    def num_data_records(self) -> int:
        """Number of data records in the recording."""
        return int(decode_str(self._num_data_records))

    @property
    def data_record_duration(self) -> float:
        """Duration of each data record in seconds."""
        return decode_float(self._data_record_duration)

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
        if not self.signals:
            raise ValueError(
                "Data record duration must be zero for annotation-only files"
            )
        for signal in self.signals:
            spr = signal.sampling_frequency * data_record_duration
            if spr % 1:
                raise ValueError(
                    f"Cannot set data record duration to {data_record_duration}: Incompatible sampling frequency {signal.sampling_frequency} Hz"
                )

        num_data_records = self._pad_or_truncate_signals(data_record_duration, method)
        self._update_record_duration_in_annotation_signals(
            data_record_duration, num_data_records
        )
        self._set_data_record_duration(data_record_duration)
        self._set_num_data_records(num_data_records)

    @property
    def _total_num_signals(self) -> int:
        return int(decode_str(self._num_signals))

    @property
    def num_signals(self) -> int:
        """Return the number of signals, excluding annotation signals for EDF+."""
        return len(self.signals)

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
        signals = list(self._signals)
        for idx, signal in enumerate(self._signals):
            if signal not in self._annotation_signals:
                continue
            annotations = []
            for data_record in signal.digital.reshape(
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
            n_samples = round(new_duration * signal.sampling_frequency)
            diff = n_samples - len(signal.digital)
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
        i = 0
        for signal in self._signals:
            if signal.label == "EDF Annotations":
                selected.append(signal)
                continue
            if i in drop or signal.label in drop:
                dropped.append(i)
                dropped.append(signal.label)
            else:
                selected.append(signal)
            i += 1
        if not_dropped := set(drop) - set(dropped):
            raise ValueError(f"No signal found with index/label {not_dropped}")
        self._signals = tuple(selected)
        self._set_bytes_in_header_record(256 * (len(selected) + 1))
        self._set_num_signals(len(selected))

    def append_signals(self, new_signals: EdfSignal | Iterable[EdfSignal]) -> None:
        """
        Append one or more signal(s) to the Edf recording.

        Every signal must be compatible with the current `data_record_duration` and all
        signal durations must match the overall recording duration. For recordings
        containing EDF+ annotation signals, the new signals are inserted after the last
        ordinary (i.e. non-annotation) signal.

        Parameters
        ----------
        new_signals : EdfSignal | Iterable[EdfSignal]
            The signal(s) to add.
        """
        if isinstance(new_signals, EdfSignal):
            new_signals = [new_signals]
        last_ordinary_index = 0
        for i, signal in enumerate(self._signals):
            if signal.label != "EDF Annotations":
                last_ordinary_index = i
        self._set_signals(
            [
                *self._signals[: last_ordinary_index + 1],
                *new_signals,
                *self._signals[last_ordinary_index + 1 :],
            ]
        )

    @property
    def _annotation_signals(self) -> Iterable[EdfSignal]:
        return (signal for signal in self._signals if signal.label == "EDF Annotations")

    @property
    def _timekeeping_signal(self) -> EdfSignal:
        return next(iter(self._annotation_signals))

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        return self.num_data_records * self.data_record_duration

    def get_annotations(
        self, start_second: float | None = None, stop_second: float | None = None
    ) -> tuple[EdfAnnotation, ...]:
        """
        All annotations defined (starting) in a specified time region of the Edf, sorted chronologically.

        Does not include timekeeping annotations. Will not include annotations that start within the specified time region but are
        defined in data records outside of that window.

        Parameters
        ----------
        start_second : float, optional
            The start of the time region in seconds from recording start. If not provided, the start of the recording is used.
        stop_second : float, optional
            The end of the time region in seconds. If not provided, the end of the recording is used.
        """
        annotations: list[EdfAnnotation] = []
        start_second_defined = start_second is not None
        stop_second_defined = stop_second is not None
        start_second = start_second or 0
        stop_second = stop_second or self.duration
        for i, signal in enumerate(self._annotation_signals):
            if self.duration == 0 or signal.sampling_frequency == 0:
                digital_slice = signal.digital
            else:
                # Make sure to read full data records
                start_record = start_second // self.data_record_duration
                adjusted_start = start_record * self.data_record_duration
                last_record = ceil(
                    stop_second / self.data_record_duration - 1e-12
                )  # avoid floating point errors
                adjusted_stop = last_record * self.data_record_duration
                digital_slice = signal.get_digital_slice(adjusted_start, adjusted_stop)
            for data_record in digital_slice.reshape(
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
        annotations = sorted(annotations)
        if start_second_defined:
            while annotations and annotations[0].onset < start_second:
                annotations.pop(0)
        if stop_second_defined:
            while annotations and annotations[-1].onset >= stop_second:
                annotations.pop()
        return tuple(annotations)

    @property
    def annotations(self) -> tuple[EdfAnnotation, ...]:
        """
        All annotations contained in the Edf, sorted chronologically.

        Does not include timekeeping annotations.
        """
        return self.get_annotations()

    def drop_annotations(self, text: str) -> None:
        """
        Drop annotations with a given text.

        Parameters
        ----------
        text : str
            All annotations whose text exactly matches this parameter are removed.
        """
        for signal in self._annotation_signals:
            for data_record in signal.digital.reshape(
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

    def set_annotations(self, annotations: Iterable[EdfAnnotation]) -> None:
        """
        Overwrite all annotations with new ones.

        This removes all existing annotation signals and adds a new one as the last
        signal in the file.

        Parameters
        ----------
        annotations : Iterable[EdfAnnotation]
            The annotations to set.
        """
        new_annotation_signal = _create_annotations_signal(
            annotations,
            num_data_records=self.num_data_records,
            data_record_duration=self.data_record_duration,
            subsecond_offset=self.starttime.microsecond / 1_000_000,
        )
        self._set_signals((*self.signals, new_annotation_signal))

    def add_annotations(self, annotations: Iterable[EdfAnnotation]) -> None:
        """
        Add annotations to the Edf.

        This removes all existing annotation signals and adds a new one containing the
        old and new annotations as the last signal in the file.

        Parameters
        ----------
        annotations : Iterable[EdfAnnotation]
            The annotations to add.
        """
        self.set_annotations(self.annotations + tuple(annotations))

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
        self._set_num_data_records(int((stop - start) / self.data_record_duration))
        for signal in self._signals:
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
                signal._digital = signal.digital[int(start_index) : int(stop_index)]
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

    def get_datarecord_onset_datetimes(self) -> list[datetime.datetime]:
        """
        Get data record onset datetimes for EDF+D discontinuous files.

        This method extracts the relative onset times from the timekeeping annotation signal
        and returns them as absolute datetime objects for each data record onset. This is
        particularly useful for EDF+D files where data records may not be
        temporally contiguous.

        Returns
        -------
        list[datetime.datetime]
            List of datetime objects representing the absolute time of each
            data record onset.

        """
        # Calculate starting datetime
        start_datetime = datetime.datetime.combine(self.startdate, self.starttime)
        # Convert relative onset times to absolute datetimes
        onset_times = self.get_datarecord_relative_onset_times()
        return [
            start_datetime + datetime.timedelta(seconds=onset_time)
            for onset_time in onset_times
        ]

    def get_datarecord_relative_onset_times(
        self, *, use_cache: bool = True
    ) -> NDArray[np.float64]:
        """
        Get relative onset times for EDF+D discontinuous files.

        This method extracts the relative onset times from the timekeeping annotation signal.
        Results are cached for performance on subsequent calls unless disabled.

        Parameters
        ----------
        use_cache : bool, default: True
            If True, uses cached results if available and caches new results.
            If False, always recalculates from the raw data.

        Returns
        -------
        NDArray[np.float64]
            Array of relative onset times in seconds from the recording start.

        Raises
        ------
        ValueError
            If the EDF file is not discontinuous (EDF+D) or lacks annotation signals.

        Examples
        --------
        >>> import edfio
        >>> edf = edfio.read_edf("discontinuous.edf")
        >>> onset_times = edf.get_relative_onset_times()
        >>> print(f"First onset at: {onset_times[0]} seconds")
        """
        if use_cache and self._cached_onset_times is not None:
            return self._cached_onset_times

        onset_times = self._compute_relative_onset_times()

        if use_cache:
            # Cache results if the number of data records is reasonable
            # Avoid caching for very large files to prevent memory issues
            cache_threshold = 100_000
            if len(onset_times) < cache_threshold:
                self._cached_onset_times = onset_times

        return onset_times

    def _compute_relative_onset_times(self) -> NDArray[np.float64]:
        """
        Compute relative onset times from the timekeeping annotation signal.

        This is the core computation method that extracts onset times from the
        EDF+D TAL (Time-stamped Annotation List) format.

        Returns
        -------
        NDArray[np.float64]
            Array of relative onset times in seconds from the recording start.

        Raises
        ------
        ValueError
            If the EDF file is not discontinuous (EDF+D) or lacks annotation signals.
        """
        if not self.is_discontinuous:
            raise ValueError(
                "Onset time map is only available for discontinuous EDF+D files"
            )

        try:
            timekeeping_signal = self._timekeeping_signal
        except StopIteration as err:
            raise ValueError("No timekeeping signal found in EDF file") from err

        # Extract onset times from the timekeeping signal
        onset_times = []
        for data_record in timekeeping_signal.digital.reshape(
            (-1, timekeeping_signal.samples_per_data_record)
        ):
            # Convert digital data to bytes and parse the first TAL
            tal_bytes = data_record.tobytes()

            # Find the end of the onset time (marked by 0x14 byte per EDF+ TAL spec)
            # Extract and parse the datarecord TAL onset time has form "[+-](\d+(\.\d*)?|\.\d+)"
            # ends with 0x14 0x14, probably better to use a compiled regular expression
            # than a loop in python
            onset_end = 0
            while onset_end < len(tal_bytes) and tal_bytes[onset_end] != 20:  # noqa: PLR2004
                onset_end += 1

            if onset_end == 0 or onset_end >= len(tal_bytes):
                continue

            try:
                # Extract and parse the datarecord TAL onset time has form "[+-](\d+(\.\d*)?|\.\d+)"
                onset_str = tal_bytes[:onset_end].decode("ascii")
                onset_seconds = float(onset_str)
                onset_times.append(onset_seconds)
            except (ValueError, UnicodeDecodeError):
                continue

        return np.array(onset_times, dtype=np.float64)

    def get_datarecord_discontinuity_indices(self) -> tuple(
        NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]
    ):
        """
        Find data record indices and timejumps where discontinuities occur in EDF+D files.

        This method analyzes the relative onset times to identify where temporal
        discontinuities occur (i.e., gaps larger than the expected data record duration).
        Uses the approach from the example code with np.diff and np.where.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing three arrays:
            - indices: NDArray[np.int64]
                Array of data record indices where discontinuities occur.
                These are indices into the data record array where a temporal jump happens.
            - time_jumps: NDArray[np.float64]
                Array of times of how much the time jumps forward
            - relative_times: NDArray[np.float64]
                The times of the discontinuities

        Raises
        ------
        ValueError
            If the EDF file is not discontinuous (EDF+D) or lacks annotation signals.

        Examples
        --------
        >>> import edfio
        >>> edf = edfio.read_edf("discontinuous.edf")
        >>> disc_indices, time_jumps, rel_times = edf.get_datarecord_discontinuity_indices()
        >>> print(f"Discontinuities at data records: {disc_indices}")
        >>> print(f"Jump are: {time_jumps} at {rel_times")
        """
        relative_times = self.get_datarecord_relative_onset_times()

        # Calculate differences between consecutive onset times
        time_jumps = np.diff(relative_times)

        # Find where differences are significantly larger than expected
        # Use data_record_duration + small tolerance as threshold
        threshold = self.data_record_duration * 1.1  # 10% tolerance
        indices = np.where(time_jumps > threshold)[0]
        return indices, time_jumps[indices], relative_times[indices]

    def _get_data_record_range(
        self, start_samp: int, end_samp: int, extra_samp: int = 0
    ) -> tuple[int, int]:
        """
        Get data record indices that contain the specified sample range.

        This function converts a sample range [sample1 - extra, sample2 + extra)
        to the corresponding data record indices that contain those samples.
        Based on the algorithm from get_annotations method.

        Parameters
        ----------
        start_samp : int
            Start sample index (must be non-negative).
        end_samp : int
            End sample index (must be non-negative and >= start_samp).
        extra_samp : int, default: 0
            Extra samples to include before start_samp and after end_samp (must be non-negative).

        Returns
        -------
        tuple[int, int]
            A tuple (start_record, end_record) where:
            - start_record: First data record index containing the range
            - end_record: Last data record index containing the range + 1 (exclusive)
            So data records [start_record, end_record) contain the sample range.

        Raises
        ------
        ValueError
            If any argument is negative or if end_samp < start_samp.

        Examples
        --------
        >>> import edfio
        >>> edf = edfio.read_edf("file.edf")
        >>> # Get data records containing samples 1000-2000 with 100 sample buffer
        >>> start_rec, end_rec = edf.get_data_record_range(1000, 2000, 100)
        >>> print(f"Need data records {start_rec} to {end_rec-1}")
        """
        # Validate inputs, probably will remove these checks
        if start_samp < 0:
            raise ValueError("start_samp must be non-negative")
        if end_samp < 0:
            raise ValueError("end_samp must be non-negative")
        if extra_samp < 0:
            raise ValueError("extra_samp must be non-negative")
        if end_samp < start_samp:
            raise ValueError("end_samp must be >= start_samp")

        # Get the first signal to determine sampling frequency
        # All signals in an EDF should have consistent timing relative to data records
        if not self.signals:
            raise ValueError("No signals available for data record calculation")

        signal = self.signals[0]

        # Calculate samples per data record (may not be set for new EDF objects)
        try:
            samples_per_record = signal.samples_per_data_record
        except AttributeError:
            # For newly created EDF objects, calculate from signal data and num_data_records
            samples_per_record = len(signal.digital) // self.num_data_records

        total_samples = samples_per_record * self.num_data_records

        # Calculate actual sample range with buffer
        start_sample = max(0, start_samp - extra_samp)  # Ensure we don't go negative
        end_sample = min(end_samp + extra_samp, total_samples)

        start_record = start_sample // samples_per_record
        end_record = math.ceil(end_sample / samples_per_record)

        # Ensure we don't exceed the available data records
        if end_record > self.num_data_records:
            raise ValueError(
                f"Calculated end_record ({end_record}) exceeds available data records ({self.num_data_records})"
            )

        return start_record, end_record

    def _invalidate_onset_cache(self) -> None:
        """Invalidate the cached onset times when EDF structure changes."""
        if hasattr(self, "_cached_onset_times"):
            self._cached_onset_times = None

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
        for data_record in signal.digital.reshape((-1, signal.samples_per_data_record)):
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
def _read_edf(edf_file: Any, *, lazy_load_data: bool, header_encoding: str) -> Edf:
    edf = object.__new__(Edf)
    edf._read_header(edf_file, header_encoding)
    edf._load_data(edf_file, lazy_load_data=lazy_load_data)
    return edf


@_read_edf.register
def _(edf_file: Path, *, lazy_load_data: bool, header_encoding: str) -> Edf:
    edf = object.__new__(Edf)
    edf_file = edf_file.expanduser()
    with edf_file.open("rb") as file:
        edf._read_header(file, header_encoding)
    edf._load_data(edf_file, lazy_load_data=lazy_load_data)
    return edf


@_read_edf.register
def _(edf_file: str, *, lazy_load_data: bool, header_encoding: str) -> Edf:
    return _read_edf(
        Path(edf_file),
        lazy_load_data=lazy_load_data,
        header_encoding=header_encoding,
    )


@_read_edf.register
def _(edf_file: bytes, *, lazy_load_data: bool, header_encoding: str) -> Edf:
    return _read_edf(
        io.BytesIO(edf_file),
        lazy_load_data=lazy_load_data,
        header_encoding=header_encoding,
    )


# Pyright loses information about parameters for singledispatch functions. Hiding it
# behind this normal function makes things work again.
def read_edf(
    edf_file: Path
    | str
    | io.BufferedReader
    | io.BytesIO
    | bytes
    | tempfile.SpooledTemporaryFile[bytes],
    lazy_load_data: bool | Literal["auto"] = "auto",
    *,
    header_encoding: str = "ascii",
) -> Edf:
    """
    Read an EDF file into an :class:`Edf` object.

    If a file-like object is passed, its stream position is moved to EOF.

    Parameters
    ----------
    edf_file : Path | str | io.BufferedReader | io.BytesIO
        The file location (path object or string) or file-like object to read from.
    lazy_load_data : bool | {"auto"}, default: "auto"
        If `True`, the raw signal data is not loaded into memory until it is accessed. If `False`,
        the data is loaded immediately. If `"auto"`, the data is loaded lazily if
        the specified edf_file represents a local path and eagerly otherwise.
    header_encoding : str, default: "ascii"
        The character encoding to use when reading header fields.

    Returns
    -------
    Edf
        The resulting :class:`Edf` object.
    """
    if lazy_load_data == "auto":
        lazy_load_data = isinstance(edf_file, (Path, str))
    return _read_edf(
        edf_file,
        lazy_load_data=lazy_load_data,
        header_encoding=header_encoding,
    )

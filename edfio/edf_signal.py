from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from typing import Callable, NamedTuple

import numpy as np
import numpy.typing as npt

from edfio._header_field import (
    RawHeaderFieldFloat,
    RawHeaderFieldInt,
    RawHeaderFieldStr,
)
from edfio.edf_annotations import EdfAnnotation, _EdfAnnotationsDataRecord, _EdfTAL


class _IntRange(NamedTuple):
    min: int
    max: int


class _FloatRange(NamedTuple):
    min: float
    max: float


def _round_float_to_8_characters(
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


def _calculate_gain_and_offset(
    digital_min: int,
    digital_max: int,
    physical_min: float,
    physical_max: float,
) -> tuple[float, float]:
    gain = (physical_max - physical_min) / (digital_max - digital_min)
    offset = physical_max / gain - digital_max
    return gain, offset


class AbstractEdfSignal:
    """Abstract base class for EDF signals."""

    _label = RawHeaderFieldStr(16, is_settable=True)
    _transducer_type = RawHeaderFieldStr(80, is_settable=True)
    _physical_dimension = RawHeaderFieldStr(8, is_settable=True)
    physical_min = RawHeaderFieldFloat(8)
    """Physical minimum, e.g., `-500` or `34`."""
    physical_max = RawHeaderFieldFloat(8)
    """Physical maximum, e.g., `500` or `40`."""
    digital_min = RawHeaderFieldInt(8)
    """Digital minimum, e.g., `-2048`."""
    digital_max = RawHeaderFieldInt(8)
    """Digital maximum, e.g., `2047`."""
    _prefiltering = RawHeaderFieldStr(80, is_settable=True)
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
        sampling_frequency: float,
        samples_per_data_record: int = -1,
        *,
        label: str = "",
        transducer_type: str = "",
        physical_dimension: str = "",
        reserved: str = "",
        digital_range: tuple[int, int] = (-32768, 32767),
        prefiltering: str = "",
    ):
        self._sampling_frequency = sampling_frequency
        self._label = label
        self._transducer_type = transducer_type
        self._physical_dimension = physical_dimension
        self._prefiltering = prefiltering
        self._reserved = AbstractEdfSignal.reserved.encode(reserved)
        self._samples_per_data_record = (
            AbstractEdfSignal.samples_per_data_record.encode(samples_per_data_record)
        )
        self._set_digital_range(digital_range)

    @property
    def label(self) -> str:
        """Signal label, e.g., `"EEG Fpz-Cz"` or `"Body temp"`."""
        return self._label

    @property
    def transducer_type(self) -> str:
        """Transducer type, e.g., `"AgAgCl electrode"`."""
        return self._transducer_type

    @property
    def physical_dimension(self) -> str:
        """Physical dimension, e.g., `"uV"` or `"degreeC"`."""
        return self._physical_dimension

    @property
    def prefiltering(self) -> str:
        """Signal prefiltering, e.g., `"HP:0.1Hz LP:75Hz"`."""
        return self._prefiltering

    @property
    def physical_range(self) -> _FloatRange:
        """The physical range as a tuple of `(physical_min, physical_max)`."""
        return _FloatRange(self.physical_min, self.physical_max)

    @property
    def digital_range(self) -> _IntRange:
        """The digital range as a tuple of `(digital_min, digital_max)`."""
        return _IntRange(self.digital_min, self.digital_max)

    @property
    def sampling_frequency(self) -> float:
        """The sampling frequency in Hz."""
        return self._sampling_frequency

    @classmethod
    def _from_raw_header(
        cls,
        sampling_frequency: float,
        *,
        _label: bytes,
        _transducer_type: bytes,
        _physical_dimension: bytes,
        physical_min: bytes,
        physical_max: bytes,
        digital_min: bytes,
        digital_max: bytes,
        _prefiltering: bytes,
        samples_per_data_record: bytes,
        reserved: bytes,
    ) -> EdfSignal | EdfAnnotationsSignal:
        label = AbstractEdfSignal._label.decode(_label)
        signal_type = EdfAnnotationsSignal if label == "EDF Annotations" else EdfSignal
        sig: EdfAnnotationsSignal | EdfSignal = object.__new__(signal_type)
        sig._sampling_frequency = sampling_frequency
        sig._label = label  # type: ignore[attr-defined]
        sig._transducer_type = AbstractEdfSignal._transducer_type.decode(_transducer_type)  # type: ignore[attr-defined]
        sig._physical_dimension = AbstractEdfSignal._physical_dimension.decode(_physical_dimension)  # type: ignore[attr-defined]
        sig._physical_min = physical_min  # type: ignore[attr-defined]
        sig._physical_max = physical_max  # type: ignore[attr-defined]
        sig._digital_min = digital_min  # type: ignore[attr-defined]
        sig._digital_max = digital_max  # type: ignore[attr-defined]
        sig._prefiltering = AbstractEdfSignal._prefiltering.decode(_prefiltering)  # type: ignore[attr-defined]
        sig._samples_per_data_record = samples_per_data_record  # type: ignore[attr-defined]
        sig._reserved = reserved  # type: ignore[attr-defined]
        return sig

    def _set_digital_range(self, digital_range: tuple[int, int]) -> None:
        digital_range = _IntRange(*digital_range)
        if digital_range.min == digital_range.max:
            raise ValueError(
                f"Digital minimum ({digital_range.min}) must differ from digital maximum ({digital_range.max})."
            )
        self._digital_min = EdfSignal.digital_min.encode(digital_range.min)
        self._digital_max = EdfSignal.digital_max.encode(digital_range.max)


class EdfSignal(AbstractEdfSignal):
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
        self._check_label_valid(label)
        super().__init__(
            sampling_frequency=sampling_frequency,
            label=label,
            transducer_type=transducer_type,
            physical_dimension=physical_dimension,
            reserved="",
            digital_range=digital_range,
            prefiltering=prefiltering,
        )
        if not np.all(np.isfinite(data)):
            raise ValueError("Signal data must contain only finite values")
        self._set_physical_range(physical_range, data)
        self._set_data(data)

    def _check_label_valid(self, label: str) -> None:
        if label == "EDF Annotations":
            raise ValueError("Signal label must not be 'EDF Annotations'.")

    def __repr__(self) -> str:
        info = f"{self.sampling_frequency:g}Hz"
        if self.label:
            info = f"{self.label} " + info
        return f"<EdfSignal {info}>"

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

    @AbstractEdfSignal.label.setter  # type: ignore[attr-defined]
    def label(self, label: str) -> None:
        """Signal label, e.g., `"EEG Fpz-Cz"` or `"Body temp"`."""
        self._check_label_valid(label)
        self._label = label

    @AbstractEdfSignal.transducer_type.setter  # type: ignore[attr-defined]
    def transducer_type(self, transducer_type: str) -> None:
        """Transducer type, e.g., `"AgAgCl electrode"`."""
        self._transducer_type = transducer_type

    @AbstractEdfSignal.physical_dimension.setter  # type: ignore[attr-defined]
    def physical_dimension(self, physical_dimension: str) -> None:
        """Physical dimension, e.g., `"uV"` or `"degreeC"`."""
        self._physical_dimension = physical_dimension

    @AbstractEdfSignal.prefiltering.setter  # type: ignore[attr-defined]
    def prefiltering(self, prefiltering: str) -> None:
        """Prefiltering, e.g., `"HP:0.1Hz LP:75Hz"`."""
        self._prefiltering = prefiltering

    @property
    def data(self) -> npt.NDArray[np.float64]:
        """
        Numpy array containing the physical signal values as floats.

        To simplify avoiding inconsistencies between signal data and header fields,
        individual values in the returned array can not be modified. Use
        :meth:`EdfSignal.update_data` to overwrite with new physical data.
        """
        try:
            gain, offset = _calculate_gain_and_offset(
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
        if not math.isclose(expected_length_f, round(expected_length_f), rel_tol=1e-10):
            raise ValueError(
                f"Sampling frequency of {sampling_frequency} results in non-integer number of samples ({expected_length_f})"
            )
        return round(expected_length_f)

    def _set_physical_range(
        self,
        physical_range: tuple[float, float] | None,
        data: npt.NDArray[np.float64],
    ) -> None:
        if physical_range is None:
            physical_range = _FloatRange(data.min(), data.max())
            if physical_range.min == physical_range.max:
                physical_range = _FloatRange(physical_range.min, physical_range.max + 1)
        else:
            physical_range = _FloatRange(*physical_range)
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
            _round_float_to_8_characters(physical_range.min, math.floor)
        )
        self._physical_max = EdfSignal.physical_max.encode(
            _round_float_to_8_characters(physical_range.max, math.ceil)
        )

    def _set_data(self, data: npt.NDArray[np.float64]) -> None:
        gain, offset = _calculate_gain_and_offset(
            self.digital_min,
            self.digital_max,
            self.physical_min,
            self.physical_max,
        )
        self._digital = np.round(data / gain - offset).astype(np.int16)


class EdfAnnotationsSignal(AbstractEdfSignal):
    """An EDF annotations signal.

    Header fields have fixed values for this type of signal and cannot bet set.

    Parameters
    ----------
    annotations : Iterable[EdfAnnotation]
        The annotations to include in the signal.
    num_data_records : int
        The number of data records in the EDF file.
    data_record_duration : float
        The duration of each data record in seconds.
    with_timestamps : bool, default: True
        If `True`, include timekeeping annotations in each record.
    subsecond_offset : float, default: 0
        Offset in seconds to add to the start of recording.
    """

    def __init__(
        self,
        annotations: Iterable[EdfAnnotation],
        *,
        num_data_records: int,
        data_record_duration: float,
        with_timestamps: bool = True,
        subsecond_offset: float = 0,
    ):
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
        spr = maxlen // 2

        super().__init__(
            sampling_frequency=spr / data_record_duration,
            samples_per_data_record=spr,
            label="EDF Annotations",
        )
        self._physical_min = AbstractEdfSignal.physical_min.encode(-32768)
        self._physical_max = AbstractEdfSignal.physical_max.encode(32767)
        self._digital = np.frombuffer(raw, dtype=np.int16).copy()

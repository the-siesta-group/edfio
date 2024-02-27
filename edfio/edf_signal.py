from __future__ import annotations

import math
import warnings
from typing import Callable, NamedTuple

import numpy as np
import numpy.typing as npt

from edfio._header_field import (
    RawHeaderFieldFloat,
    RawHeaderFieldInt,
    RawHeaderFieldStr,
)


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

    _label = RawHeaderFieldStr(16, is_settable=True)
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
        info = f"{self.sampling_frequency:g}Hz"
        if self.label:
            info = f"{self.label} " + info
        return f"<EdfSignal {info}>"

    @classmethod
    def _from_raw_header(
        cls,
        sampling_frequency: float,
        *,
        _label: bytes,
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
        sig._label = EdfSignal._label.decode(_label)  # type: ignore[attr-defined]
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
    def label(self) -> str:
        """Signal label, e.g., `"EEG Fpz-Cz"` or `"Body temp"`."""
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        if label == "EDF Annotations":
            raise ValueError("Ordinary signal label must not be 'EDF Annotations'.")
        self._label = label

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

    def _set_digital_range(self, digital_range: tuple[int, int]) -> None:
        digital_range = _IntRange(*digital_range)
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

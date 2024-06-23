from __future__ import annotations

import math
import warnings
from typing import Callable, NamedTuple

import numpy as np
import numpy.typing as npt

from edfio._header_field import (
    decode_float,
    decode_str,
    encode_float,
    encode_int,
    encode_str,
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

    _header_fields = (
        ("label", 16),
        ("transducer_type", 80),
        ("physical_dimension", 8),
        ("physical_min", 8),
        ("physical_max", 8),
        ("digital_min", 8),
        ("digital_max", 8),
        ("prefiltering", 80),
        ("samples_per_data_record", 8),
        ("reserved", 32),
    )

    _digital: npt.NDArray[np.int16] | None = None
    _lazy_loader: Callable[[], npt.NDArray[np.int16]] | None = None

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
        self._set_reserved("")
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
        sig._label = label
        sig._transducer_type = transducer_type
        sig._physical_dimension = physical_dimension
        sig._physical_min = physical_min
        sig._physical_max = physical_max
        sig._digital_min = digital_min
        sig._digital_max = digital_max
        sig._prefiltering = prefiltering
        sig._samples_per_data_record = samples_per_data_record
        sig._reserved = reserved
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

    def _set_samples_per_data_record(self, samples_per_data_record: int) -> None:
        self._samples_per_data_record = encode_int(samples_per_data_record, 8)

    def _set_reserved(self, reserved: str) -> None:
        self._reserved = encode_str(reserved, 32)

    @property
    def label(self) -> str:
        """Signal label, e.g., `"EEG Fpz-Cz"` or `"Body temp"`."""
        return decode_str(self._label)

    @label.setter
    def label(self, label: str) -> None:
        if label == "EDF Annotations":
            raise ValueError("Ordinary signal label must not be 'EDF Annotations'.")
        self._label = encode_str(label, 16)

    @property
    def transducer_type(self) -> str:
        """Transducer type, e.g., `"AgAgCl electrode"`."""
        return decode_str(self._transducer_type)

    @transducer_type.setter
    def transducer_type(self, transducer_type: str) -> None:
        self._transducer_type = encode_str(transducer_type, 80)

    @property
    def physical_dimension(self) -> str:
        """Physical dimension, e.g., `"uV"` or `"degreeC`."""
        return decode_str(self._physical_dimension)

    @physical_dimension.setter
    def physical_dimension(self, physical_dimension: str) -> None:
        self._physical_dimension = encode_str(physical_dimension, 8)

    @property
    def physical_min(self) -> float:
        """Physical minimum, e.g., `-500` or `34`."""
        return decode_float(self._physical_min)

    @property
    def physical_max(self) -> float:
        """Physical maximum, e.g., `500` or `40`."""
        return decode_float(self._physical_max)

    @property
    def digital_min(self) -> int:
        """Digital minimum, e.g., `-2048`."""
        return int(decode_str(self._digital_min))

    @property
    def digital_max(self) -> int:
        """Digital maximum, e.g., `2047`."""
        return int(decode_str(self._digital_max))

    @property
    def prefiltering(self) -> str:
        """Signal prefiltering, e.g., `"HP:0.1Hz LP:75Hz"`."""
        return decode_str(self._prefiltering)

    @prefiltering.setter
    def prefiltering(self, prefiltering: str) -> None:
        self._prefiltering = encode_str(prefiltering, 80)

    @property
    def samples_per_data_record(self) -> int:
        """
        Number of samples in each data record.

        For newly instantiated :class:`EdfSignal` objects, this is only set once
        :meth:`Edf.write` is called.
        """
        return int(decode_str(self._samples_per_data_record))

    @property
    def reserved(self) -> str:
        """Reserved signal header field, always `""`."""
        return decode_str(self._reserved)

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
    def digital(self) -> npt.NDArray[np.int16]:
        """
        Numpy array containing the digital (uncalibrated) signal values as 16-bit integers.

        The values of the array may be accessed and modified directly.
        """
        if self._digital is None:
            if self._lazy_loader is None:
                raise ValueError("Signal data not set")
            self._digital = self._lazy_loader()
            self._lazy_loader = None
        return self._digital

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
            data = self.digital.astype(np.float64)
            if self.digital_max == self.digital_min:
                warnings.warn(
                    f"Digital minimum equals digital maximum ({self.digital_min}) for {self.label}, returning uncalibrated signal."
                )
            else:
                warnings.warn(
                    f"Physical minimum equals physical maximum ({self.physical_min}) for {self.label}, returning uncalibrated signal."
                )
        except ValueError:
            data = self.digital.astype(np.float64)
        else:
            data = (self.digital + offset) * gain
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
        expected_length = len(self.digital)
        if (
            sampling_frequency is not None
            and sampling_frequency != self._sampling_frequency
        ):
            expected_length = self._get_expected_new_length(sampling_frequency)
        if len(data) != expected_length:
            raise ValueError(
                f"Signal lengths must match: got {len(data)}, expected {len(self.digital)}."
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
        current_length = len(self.digital)
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
        self._digital_min = encode_int(digital_range.min, 8)
        self._digital_max = encode_int(digital_range.max, 8)

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
        self._physical_min = encode_float(
            _round_float_to_8_characters(physical_range.min, math.floor)
        )
        self._physical_max = encode_float(
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

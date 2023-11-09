"""
Tests to verify the adherence to the EDF(+) programming guidelines:
https://www.edfplus.info/specs/guidelines.html
"""

import numpy as np
import pytest

from edfio._header_field import decode_float
from edfio.edf import EdfSignal


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # (b"1E2345  ", 1),  # mentioned in guidelines, but exceeds the range of double
        (b"+012E+34", 12e34),
        (b"-1.34E09", -1.34e9),
        (b"+1.23E-5", 1.23e-5),  # guidelines mention "+1.234E-5", but that has 9 chars
    ],
)
def test_g2a_float_decode_different_formats(field: bytes, value: float):
    assert decode_float(field) == value


def test_g8_edf_signal_from_hypnogram_sets_header_fields_correctly():
    stages = np.array([0, 1, 2, 3, 4, 5, 6, 9, 0])
    hypnogram = EdfSignal.from_hypnogram(stages)
    np.testing.assert_equal(stages, hypnogram.data)
    assert hypnogram.sampling_frequency == 1 / 30
    assert hypnogram.physical_range == (0, 9)
    assert hypnogram.digital_range == (0, 9)


@pytest.mark.parametrize(
    "stages",
    [
        (-1, 0, 1),
        (0, 1, 7),
        (0, 1, 8),
        (0, 1, 10),
    ],
)
def test_g8_edf_signal_from_hypnogram_fails_for_invalid_stages(stages: tuple[int]):
    with pytest.raises(ValueError, match="stages contains invalid values"):
        EdfSignal.from_hypnogram(np.array(stages))

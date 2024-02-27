import pytest

from edfio._utils import encode_annotation_duration, encode_annotation_onset


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
    assert encode_annotation_onset(onset) == expected


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
    assert encode_annotation_duration(duration) == expected


def test_encode_annotation_duration_raises_error_for_negative_values():
    with pytest.raises(ValueError, match="Annotation duration must be positive, is"):
        encode_annotation_duration(-1)

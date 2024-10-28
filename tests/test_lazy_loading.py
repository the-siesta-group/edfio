import numpy as np
import pytest

from edfio._lazy_loading import LazyLoader


def test_load_entire_signal_data(buffered_lazy_loader: LazyLoader):
    expected_data = np.arange(1, 13, dtype=np.int16)
    np.testing.assert_array_equal(buffered_lazy_loader.load(), expected_data)


def test_load_some_records(buffered_lazy_loader: LazyLoader):
    expected_data = np.arange(4, 10, dtype=np.int16)
    np.testing.assert_array_equal(buffered_lazy_loader.load(1, 3), expected_data)


@pytest.mark.parametrize(
    ("start_record", "end_record"),
    [
        (-1, 3),
        (1, 5),
        (3, 2),
    ],
)
def test_load_invalid_records(
    start_record: int, end_record: int, buffered_lazy_loader: LazyLoader
):
    with pytest.raises(ValueError, match="Invalid slice: Slice exceeds EDF duration"):
        buffered_lazy_loader.load(start_record, end_record)

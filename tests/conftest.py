from pathlib import Path

import numpy as np
import pytest

from edfio._lazy_loading import LazyLoader


@pytest.fixture()
def tmp_file(tmp_path: Path) -> Path:
    return tmp_path / "target.edf"


@pytest.fixture()
def buffered_lazy_loader() -> LazyLoader:
    # Buffer of 4 raw EDF records containing 12 samples for the tested signal.
    # Data for other signals is set to zero.
    data_records = np.array(
        [
            [0, 0, 0, 1, 2, 3, 0, 0],
            [0, 0, 0, 4, 5, 6, 0, 0],
            [0, 0, 0, 7, 8, 9, 0, 0],
            [0, 0, 0, 10, 11, 12, 0, 0],
        ],
        dtype=np.int16,
    )
    return LazyLoader(data_records, 3, 6)

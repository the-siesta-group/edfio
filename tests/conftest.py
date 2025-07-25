from pathlib import Path
from typing import Tuple, Type

import numpy as np
import pytest

from edfio import Edf, Bdf, EdfSignal, BdfSignal
from edfio._lazy_loading import LazyLoader


def pytest_configure(config):
    """Configure pytest."""
    # Set numpy print options for tests
    for marker in (
        "edf_format: mark a test as using an EDF formatting",
        "bdf_format: mark a test as using an BDF formatting",
    ):
        config.addinivalue_line("markers", marker)


@pytest.fixture
def tmp_file(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    if request.node.get_closest_marker("bdf_format") is not None:
        ext = "bdf"
    else:
        ext = "edf"

    return tmp_path / f"target.{ext}"


@pytest.fixture
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


@pytest.fixture(params=["edf", "bdf"], scope="session")
def klasses(request) -> Tuple[Type, Type]:
    """Parametrizes the name of the browser backend."""
    request.applymarker(f"{request.param}_format")
    if request.param == "edf":
        return Edf, EdfSignal, request.param
    else:
        assert request.param == "bdf"
        return Bdf, BdfSignal, request.param

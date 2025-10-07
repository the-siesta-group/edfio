import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from edfio import Bdf, BdfSignal, Edf, EdfSignal
from edfio._lazy_loading import LazyLoader
from edfio.edf import read_bdf, read_edf
from edfio.edf_signal import _BDF_DEFAULT_RANGE, _EDF_DEFAULT_RANGE


class _Context:
    format: Literal["edf", "bdf"] = "edf"
    digital_range: tuple[int, int] = _EDF_DEFAULT_RANGE
    bits: int = 16


@pytest.fixture
def tmp_file(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    return tmp_path / f"target.{_Context.format}"


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


def pytest_collection_modifyitems(config, items):
    keep = []
    for item in items:
        format = item.callspec.params["inject_classes"]
        markers = {m.name for m in item.iter_markers()}
        if markers & {"bdf", "edf"} and format not in markers:
            continue
        keep.append(item)
    items[:] = keep


@pytest.fixture(params=["edf", "bdf"], ids=["edf", "bdf"], autouse=True)
def inject_classes(request):
    formats = {
        "edf": (Edf, EdfSignal, _EDF_DEFAULT_RANGE, 16, read_edf),
        "bdf": (Bdf, BdfSignal, _BDF_DEFAULT_RANGE, 24, read_bdf),
    }
    format = request.param
    file_class, signal_class, digital_range, bits, read_func = formats[format]
    if not request.node.get_closest_marker(format):
        for name, param in request.node.callspec.params.items():
            if isinstance(param, (Edf, EdfSignal)):
                raise ValueError(
                    f"Test parameter {name!r} ({param}) breaks automatically extending the test to BDF. "
                )
    module = sys.modules[request.node.obj.__module__]
    module.Edf = file_class
    module.EdfSignal = signal_class
    module.read_edf = read_func
    _Context.digital_range = digital_range
    _Context.bits = bits
    _Context.format = format
    for file_constant in (
        "EDF_FILE",
        "MNE_TEST_FILE",
        "SUBSECOND_TEST_FILE",
        "ONLY_ANNOTATIONS_FILE",
    ):
        if path := getattr(module, file_constant, None):
            setattr(module, file_constant, path.with_suffix(f".{format}"))

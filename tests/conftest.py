from pathlib import Path

import pytest


@pytest.fixture
def tmp_file(tmp_path: Path) -> Path:
    return tmp_path / "target.edf"

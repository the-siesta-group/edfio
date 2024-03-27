# edfio

![Python](https://img.shields.io/pypi/pyversions/edfio)
[![PyPI](https://img.shields.io/pypi/v/edfio)](https://pypi.org/project/edfio/)
[![conda-forge](https://img.shields.io/conda/v/conda-forge/edfio.svg?label=conda-forge)](https://anaconda.org/conda-forge/edfio)
[![License](https://img.shields.io/pypi/l/edfio)](https://github.com/the-siesta-group/edfio/blob/main/LICENSE)
[![Docs](https://readthedocs.org/projects/edfio/badge)](https://edfio.readthedocs.io/en/stable/index.html)

[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


edfio is a Python package for reading and writing [EDF](https://www.edfplus.info/specs/edf.html) and [EDF+C](https://www.edfplus.info/specs/edfplus.html) files.

It requires Python>=3.9 and NumPy>=1.22 and is available on PyPI:

    pip install edfio

- Support for EDF and EDF+C, including annotations
- Fast I/O thanks to NumPy - read and write GB-sized files in seconds
- Fail late on read: Non-compliant header fields only raise an exception when the corresponding property is accessed.
- Fail early on write: Trying to create a new non-compliant EDF file raises an exception.
- Object-oriented design and type annotations for IDE autocompletion
- Pure Python implementation and 100% test coverage to simplify contributions


## Features
- Read/write from/to files or file-like objects
- Modify signal and recording headers
- Drop EDF+ annotations
- Slice recordings (by seconds or annotation texts)
- Drop individual signals
- Anonymize recordings


## Known limitations
- Discontiguous files (EDF+D) are treated as contiguous ones.
- The maximum data record size of 61440 bytes recommended by the [EDF specs](https://www.edfplus.info/specs/edf.html) is not enforced.
- To write an EDF with a non-integer seconds duration, the data record duration has to be manually set to an appropriate value.
- Slicing an EDF to a timespan that is not an integer multiple of the data record duration does not work.
- BDF files ([BioSemi](https://www.biosemi.com/faq/file_format.htm)) are not supported.


## Contributing
Contributions are welcome and highly appreciated.
Check out the [contributing guidelines](https://edfio.readthedocs.io/en/stable/contributing) to get started.


## Usage
Further information is available in the [API reference](https://edfio.readthedocs.io/en/stable/reference) and [usage examples](https://edfio.readthedocs.io/en/stable/examples).

To read an EDF from a file, use `edfio.read_edf`:

```python
from edfio import read_edf

edf = read_edf("example.edf")
```

A new EDF can be created and written to a file as follows:

```python
import numpy as np

from edfio import Edf, EdfSignal

edf = Edf(
    [
        EdfSignal(np.random.randn(30 * 256), sampling_frequency=256, label="EEG Fpz"),
        EdfSignal(np.random.randn(30), sampling_frequency=1, label="Body Temp"),
    ]
)
edf.write("example.edf")
```

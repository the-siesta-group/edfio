from edfio.edf import Edf, read_edf
from edfio.edf_annotations import EdfAnnotation
from edfio.edf_header import AnonymizedDateError, Patient, Recording
from edfio.edf_signal import EdfSignal

try:
    from importlib.metadata import version

    __version__ = version("edfio")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"


__all__ = [
    "AnonymizedDateError",
    "Edf",
    "EdfAnnotation",
    "EdfSignal",
    "Patient",
    "Recording",
    "read_edf",
]

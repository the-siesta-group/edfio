from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from edfio.edf_signal import EdfSignal

_ANNOTATIONS_PATTERN = re.compile(
    """
    ([+-]\\d+(?:\\.?\\d+)?)       # onset
    (?:\x15(\\d+(?:\\.?\\d+)?))?  # duration, optional
    (?:\x14(.*?))                 # annotation texts
    \x14\x00                      # terminator
    """,
    re.VERBOSE,
)


def _encode_annotation_onset(onset: float) -> str:
    string = f"{onset:+.12f}".rstrip("0")
    if string[-1] == ".":
        return string[:-1]
    return string


def _encode_annotation_duration(duration: float) -> str:
    if duration < 0:
        raise ValueError(f"Annotation duration must be positive, is {duration}")
    string = f"{duration:.12f}".rstrip("0")
    if string[-1] == ".":
        return string[:-1]
    return string


class EdfAnnotation(NamedTuple):
    """A single EDF+ annotation.

    Parameters
    ----------
    onset : float
        The annotation onset in seconds from recording start.
    duration : float | None
        The annotation duration in seconds (`None` if annotation has no duration).
    text : str
        The annotation text, can be empty.
    """

    onset: float
    duration: float | None
    text: str


def _create_annotations_signal(
    annotations: Iterable[EdfAnnotation],
    *,
    num_data_records: int,
    data_record_duration: float,
    with_timestamps: bool = True,
    subsecond_offset: float = 0,
) -> EdfSignal:
    data_record_starts = np.arange(num_data_records) * data_record_duration
    annotations = sorted(annotations)
    data_records = []
    for i, start in enumerate(data_record_starts):
        end = start + data_record_duration
        tals: list[_EdfTAL] = []
        if with_timestamps:
            tals.append(_EdfTAL(np.round(start + subsecond_offset, 12), None, [""]))
        for ann in annotations:
            if (
                (i == 0 and ann.onset < 0)
                or (i == (num_data_records - 1) and end <= ann.onset)
                or (start <= ann.onset < end)
            ):
                tals.append(
                    _EdfTAL(
                        np.round(ann.onset + subsecond_offset, 12),
                        ann.duration,
                        [ann.text],
                    )
                )
        data_records.append(_EdfAnnotationsDataRecord(tals).to_bytes())
    maxlen = max(len(data_record) for data_record in data_records)
    if maxlen % 2:
        maxlen += 1
    raw = b"".join(dr.ljust(maxlen, b"\x00") for dr in data_records)
    divisor = data_record_duration if data_record_duration else 1
    signal = EdfSignal(
        np.arange(1.0),  # placeholder signal, as argument `data` is non-optional
        sampling_frequency=maxlen // 2 / divisor,
        physical_range=(-32768, 32767),
    )
    signal._label = b"EDF Annotations "
    signal._set_samples_per_data_record(maxlen // 2)
    signal._digital = np.frombuffer(raw, dtype=np.int16).copy()
    return signal


@dataclass
class _EdfTAL:
    onset: float
    duration: float | None
    texts: list[str]

    def to_bytes(self) -> bytes:
        timing = _encode_annotation_onset(self.onset)
        if self.duration is not None:
            timing += f"\x15{_encode_annotation_duration(self.duration)}"
        texts_joined = "\x14".join(self.texts)
        return f"{timing}\x14{texts_joined}\x14".encode()


@dataclass
class _EdfAnnotationsDataRecord:
    tals: list[_EdfTAL]

    def to_bytes(self) -> bytes:
        return b"\x00".join(tal.to_bytes() for tal in self.tals) + b"\x00"

    @classmethod
    def from_bytes(cls, raw: bytes) -> _EdfAnnotationsDataRecord:
        tals: list[_EdfTAL] = []
        matches: list[tuple[str, str, str]] = _ANNOTATIONS_PATTERN.findall(raw.decode())
        if not matches and raw.replace(b"\x00", b""):
            raise ValueError(f"No valid annotations found in {raw!r}")
        for onset, duration, texts in matches:
            tals.append(
                _EdfTAL(
                    float(onset),
                    float(duration) if duration else None,
                    list(texts.split("\x14")),
                )
            )
        return cls(tals)

    @property
    def annotations(self) -> list[EdfAnnotation]:
        return [
            EdfAnnotation(tal.onset, tal.duration, text)
            for tal in self.tals
            for text in tal.texts
        ]

    def drop_annotations_with_text(self, text: str) -> None:
        for tal in self.tals:
            while text in tal.texts:
                tal.texts.remove(text)
        self.tals = [tal for tal in self.tals if tal.texts]

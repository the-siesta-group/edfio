from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

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

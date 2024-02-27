from __future__ import annotations


def encode_annotation_onset(onset: float) -> str:
    string = f"{onset:+.12f}".rstrip("0")
    if string[-1] == ".":
        return string[:-1]
    return string


def encode_annotation_duration(duration: float) -> str:
    if duration < 0:
        raise ValueError(f"Annotation duration must be positive, is {duration}")
    string = f"{duration:.12f}".rstrip("0")
    if string[-1] == ".":
        return string[:-1]
    return string

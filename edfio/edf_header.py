from __future__ import annotations

import datetime
import inspect
from collections.abc import Sequence
from typing import Any, Literal

from edfio._header_field import encode_str


def _repr_from_init(obj: Any) -> str:
    parameters = []
    for name in inspect.signature(obj.__class__).parameters:
        parameters.append(f"{name}={getattr(obj, name)!r}")
    return f"{obj.__class__.__name__}({', '.join(parameters)})"


_MONTH_NAMES = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)


def _decode_edfplus_date(date: str) -> datetime.date:
    day, month, year = date.split("-")
    try:
        month_int = _MONTH_NAMES.index(month.upper()) + 1
    except ValueError:
        raise ValueError(f"Invalid month: {month}, options: {_MONTH_NAMES}") from None
    return datetime.date(int(year), month_int, int(day))


def _encode_edfplus_date(date: datetime.date) -> str:
    return f"{date.day:02}-{_MONTH_NAMES[date.month - 1]}-{date.year:02}"


def _validate_subfields(subfields: dict[str, str]) -> None:
    for key, value in subfields.items():
        if not value:
            raise ValueError(f"Subfield {key} must not be an empty string")
        if " " in value:
            raise ValueError(f"Subfield {key} contains spaces: {value!r}")


class AnonymizedDateError(ValueError):
    """Raised when trying to access an anonymized startdate or birthdate."""


class Patient:
    """
    Object representation of the local patient identification.

    Parsing from/to the string containing the local_patient_identification header field
    is done according to the EDF+ specs. Subfields must be ASCII (32..126) and may not
    contain spaces.

    Parameters
    ----------
    code : str, default: `"X"`
        The code by which the patient is known in the hospital administration.
    sex : `{"X", "F", "M"}`, default: `"X"`
        Sex, `F` for female, `M` for male, `X` if anonymized.
    birthdate : datetime.date | None, default: None
        Patient birthdate, stored as `X` if `None`.
    name : str, default: `"X"`
        The patient's name, stored as `X` if `None`.
    additional : Sequence[str], default: `()`
        Optional additional subfields. Will be stored in the header field separated by
        spaces.
    """

    def __init__(
        self,
        *,
        code: str = "X",
        sex: Literal["F", "M", "X"] = "X",
        birthdate: datetime.date | None = None,
        name: str = "X",
        additional: Sequence[str] = (),
    ) -> None:
        if sex not in ("F", "M", "X"):
            raise ValueError(f"Invalid sex: {sex}, must be one of F, M, X")
        if birthdate is None:
            birthdate_field = "X"
        else:
            birthdate_field = _encode_edfplus_date(birthdate)
        subfields = {
            "code": code,
            "sex": sex,
            "birthdate": birthdate_field,
            "name": name,
            **{f"additional[{i}]": v for i, v in enumerate(additional)},
        }
        _validate_subfields(subfields)
        local_patient_identification = " ".join(subfields.values())
        encode_str(local_patient_identification, 80)
        self._local_patient_identification = local_patient_identification

    def __repr__(self) -> str:
        try:
            return _repr_from_init(self)
        except Exception:
            return repr(self._local_patient_identification)

    @classmethod
    def _from_str(cls, string: str) -> Patient:
        encode_str(string, 80)
        obj = object.__new__(cls)
        obj._local_patient_identification = string
        return obj

    def _to_str(self) -> str:
        return self._local_patient_identification

    @property
    def code(self) -> str:
        """The code by which the patient is known in the hospital administration."""
        return self.get_subfield(0)

    @property
    def sex(self) -> str:
        """Sex, `F` for female, `M` for male, `X` if anonymized."""
        return self.get_subfield(1)

    @property
    def birthdate(self) -> datetime.date:
        """Patient birthdate."""
        birthdate_field = self.get_subfield(2)
        if birthdate_field == "X":
            raise AnonymizedDateError("Patient birthdate is not available ('X').")
        return _decode_edfplus_date(birthdate_field)

    @property
    def name(self) -> str:
        """The patient's name."""
        return self.get_subfield(3)

    @property
    def additional(self) -> tuple[str, ...]:
        """Optional additional subfields."""
        return tuple(self._local_patient_identification.split()[4:])

    def get_subfield(self, idx: int) -> str:
        """
        Access a subfield of the local patient identification field by index.

        Parameters
        ----------
        idx : int
            The index of the subfield to access.

        Returns
        -------
        str
            The subfield at the specified index. If the index exceeds the actually
            available number of subfields, the return value is `"X"`.
        """
        subfields = self._local_patient_identification.split()
        if len(subfields) <= idx:
            return "X"
        return subfields[idx]


class Recording:
    """
    Object representation of the local recording identification.

    Parsing from/to the string containing the local_recording_identification header
    field is done according to the EDF+ specs. Subfields must be ASCII (32..126) and may
    not contain spaces.

    Parameters
    ----------
    startdate : datetime.date | None, default: None
        The recording startdate.
    hospital_administration_code : str, default: `"X"`
        The hospital administration code of the investigation, e.g., EEG number or PSG
        number.
    investigator_technician_code : str, default: `"X"`
        A code specifying the responsible investigator or technician.
    equipment_code : str, default: `"X"`
        A code specifying the used equipment.
    additional : Sequence[str], default: `()`
        Optional additional subfields. Will be stored in the header field separated by
        spaces.
    """

    def __init__(
        self,
        *,
        startdate: datetime.date | None = None,
        hospital_administration_code: str = "X",
        investigator_technician_code: str = "X",
        equipment_code: str = "X",
        additional: Sequence[str] = (),
    ) -> None:
        if startdate is None:
            startdate_field = "X"
        else:
            startdate_field = _encode_edfplus_date(startdate)
        subfields = {
            "startdate": startdate_field,
            "hospital_administration_code": hospital_administration_code,
            "investigator_technician_code": investigator_technician_code,
            "equipment_code": equipment_code,
            **{f"additional[{i}]": v for i, v in enumerate(additional)},
        }
        _validate_subfields(subfields)
        local_recording_identification = " ".join(("Startdate", *subfields.values()))
        encode_str(local_recording_identification, 80)
        self._local_recording_identification = local_recording_identification

    def __repr__(self) -> str:
        try:
            return _repr_from_init(self)
        except Exception:
            return repr(self._local_recording_identification)

    @classmethod
    def _from_str(cls, string: str) -> Recording:
        encode_str(string, 80)
        obj = object.__new__(cls)
        obj._local_recording_identification = string
        return obj

    def _to_str(self) -> str:
        return self._local_recording_identification

    @property
    def startdate(self) -> datetime.date:
        """The recording startdate."""
        if not self._local_recording_identification.startswith("Startdate "):
            raise ValueError(
                f"Local recording identification field {self._local_recording_identification!r} does not follow EDF+ standard."
            )
        startdate_field = self.get_subfield(1)
        if startdate_field == "X":
            raise AnonymizedDateError("Recording startdate is not available ('X').")
        return _decode_edfplus_date(startdate_field)

    @property
    def hospital_administration_code(self) -> str:
        """The hospital administration code of the investigation."""
        return self.get_subfield(2)

    @property
    def investigator_technician_code(self) -> str:
        """A code specifying the responsible investigator or technician."""
        return self.get_subfield(3)

    @property
    def equipment_code(self) -> str:
        """A code specifying the used equipment."""
        return self.get_subfield(4)

    @property
    def additional(self) -> tuple[str, ...]:
        """Optional additional subfields."""
        return tuple(self._local_recording_identification.split()[5:])

    def get_subfield(self, idx: int) -> str:
        """
        Access a subfield of the local recording identification field by index.

        Parameters
        ----------
        idx : int
            The index of the subfield to access. The first subfield (starting at
            index 0) should always be "Startdate" according to the EDF+ spedification.

        Returns
        -------
        str
            The subfield at the specified index. If the index exceeds the actually
            available number of subfields, the return value is `"X"`.
        """
        subfields = self._local_recording_identification.split()
        if len(subfields) <= idx:
            return "X"
        return subfields[idx]

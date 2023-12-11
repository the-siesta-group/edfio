import datetime

import numpy as np
import pytest

from edfio import AnonymizedDateError, Edf, EdfSignal, Patient, Recording


@pytest.fixture()
def patient():
    return Patient._from_str("MCH-0234567 F 02-MAY-1951 Haagse_Harry")


@pytest.fixture()
def recording():
    return Recording._from_str(
        "Startdate 02-MAR-2002 EMG561 BK/JOP Sony. MNC R Median Nerve."
    )


@pytest.fixture()
def edf(patient, recording):
    return Edf([EdfSignal(np.arange(10), 1)], patient=patient, recording=recording)


LOCAL_RECORDING_IDENTIFICATION_WITH_INVALID_STARTDATE = [
    b"",
    b"xxx 01-JAN-2001 X X X",
    b"01-JAN-2001",
]


def test_patient_from_str(patient: Patient):
    assert patient.code == "MCH-0234567"
    assert patient.sex == "F"
    assert patient.birthdate == datetime.date(1951, 5, 2)
    assert patient.name == "Haagse_Harry"
    assert patient.additional == ()


def test_patient_default_init():
    assert Patient()._to_str() == "X X X X"


def test_patient_is_immutable():
    with pytest.raises(AttributeError, match="can't set attribute|has no setter"):
        Patient().code = "123"


def test_instantiating_patient_with_invalid_sex_raises_error():
    with pytest.raises(ValueError, match="Invalid sex"):
        Patient(sex="W")


def test_instantiating_patient_with_valid_birthdate():
    patient = Patient(birthdate=datetime.date(1951, 5, 2))
    assert patient._to_str() == "X X 02-MAY-1951 X"


def test_patient_birthdate_raises_error_if_not_available():
    patient = Patient._from_str("X X X X")
    with pytest.raises(AnonymizedDateError, match="birthdate is not available"):
        patient.birthdate


@pytest.mark.parametrize("birthdate", ["01-Nov-2023", "01-nov-2023", "01-noV-2023"])
def test_patient_birthdate_incorrect_case(birthdate: str):
    patient = Patient._from_str(f"X X {birthdate} X")
    assert patient.birthdate == datetime.date(2023, 11, 1)


def test_patient_allows_accessing_other_fields_if_birthdate_is_invalid():
    patient = Patient._from_str("X F 1-2-3 X")
    assert patient.sex == "F"


def test_instantiating_patient_with_subfield_containing_spaces_raises_error():
    with pytest.raises(ValueError, match="contains spaces"):
        Patient(name="Haagse Harry")


def test_patient_repr(patient: Patient):
    assert (
        repr(patient)
        == "Patient(code='MCH-0234567', sex='F', birthdate=datetime.date(1951, 5, 2), name='Haagse_Harry', additional=())"
    )


def test_patient_repr_with_invalid_birthdate():
    patient_field = "MCH-0234567 F X Haagse_Harry"
    patient = Patient._from_str(patient_field)
    assert repr(patient) == repr(patient_field)


def test_edf_patient_getter(edf: Edf):
    assert edf.local_patient_identification == "MCH-0234567 F 02-MAY-1951 Haagse_Harry"
    assert edf.patient.code == "MCH-0234567"
    assert edf.patient.sex == "F"
    assert edf.patient.birthdate == datetime.date(1951, 5, 2)
    assert edf.patient.name == "Haagse_Harry"
    assert edf.patient.additional == ()


def test_edf_patient_setter(edf: Edf):
    edf.patient = Patient()
    assert edf.local_patient_identification == "X X X X"


def test_patient_raises_error_if_too_long_for_header_field():
    with pytest.raises(ValueError, match="exceeds maximum field length"):
        Patient(code="X" * 81)


def test_patient_from_str_raises_error_if_too_long_for_header_field():
    with pytest.raises(ValueError, match="exceeds maximum field length"):
        Patient._from_str("X" * 81)


def test_patient_raises_error_on_invalid_characters():
    with pytest.raises(UnicodeEncodeError, match="'ascii' codec can't encode"):
        Patient(name="ÄÖÜ")


def test_patient_from_str_raises_error_on_invalid_characters():
    with pytest.raises(UnicodeEncodeError, match="'ascii' codec can't encode"):
        Patient._from_str("X X X AÖÜ")


def test_recording_from_str(recording: Recording):
    assert recording.startdate == datetime.date(2002, 3, 2)
    assert recording.hospital_administration_code == "EMG561"
    assert recording.investigator_technician_code == "BK/JOP"
    assert recording.equipment_code == "Sony."
    assert recording.additional == ("MNC", "R", "Median", "Nerve.")


def test_recording_default_init():
    assert Recording()._to_str() == "Startdate X X X X"


def test_recording_is_immutable():
    with pytest.raises(AttributeError, match="can't set attribute|has no setter"):
        Recording().equipment_code = "123"


def test_instantiating_recording_with_valid_startdate():
    recording = Recording(startdate=datetime.date(2002, 3, 2))
    assert recording._to_str() == "Startdate 02-MAR-2002 X X X"


def test_recording_startdate_raises_error_if_not_available():
    recording = Recording._from_str("Startdate X X X X")
    with pytest.raises(AnonymizedDateError, match="startdate is not available"):
        recording.startdate


@pytest.mark.parametrize(
    "local_recording_identification",
    LOCAL_RECORDING_IDENTIFICATION_WITH_INVALID_STARTDATE,
)
def test_recording_startdate_raises_error_on_invalid_recording_field(
    local_recording_identification: bytes,
):
    recording = Recording._from_str(local_recording_identification.decode())
    with pytest.raises(ValueError, match="does not follow EDF\\+ standard"):
        recording.startdate


@pytest.mark.parametrize("startdate", ["01-Nov-2023", "01-nov-2023", "01-noV-2023"])
def test_read_startdate_incorrect_case(startdate: str):
    recording = Recording._from_str(f"Startdate {startdate} X X X")
    assert recording.startdate == datetime.date(2023, 11, 1)


def test_recording_allows_accessing_other_fields_if_startdate_is_invalid():
    recording = Recording._from_str("Startdate 1-2-3 EMG561 X X")
    assert recording.hospital_administration_code == "EMG561"


def test_instantiating_recording_with_subfield_containing_spaces_raises_error():
    with pytest.raises(ValueError, match="contains spaces"):
        Recording(investigator_technician_code="BK JOP")


def test_recording_repr(recording: Recording):
    assert (
        repr(recording)
        == "Recording(startdate=datetime.date(2002, 3, 2), hospital_administration_code='EMG561', investigator_technician_code='BK/JOP', equipment_code='Sony.', additional=('MNC', 'R', 'Median', 'Nerve.'))"
    )


def test_recording_repr_with_invalid_startdate():
    recording_field = "Startdate X EMG561 BK/JOP Sony."
    recording = Recording._from_str(recording_field)
    assert repr(recording) == repr(recording_field)


def test_edf_recording_getter(edf: Edf):
    assert (
        edf.local_recording_identification
        == "Startdate 02-MAR-2002 EMG561 BK/JOP Sony. MNC R Median Nerve."
    )
    assert edf.recording.startdate == datetime.date(2002, 3, 2)
    assert edf.recording.hospital_administration_code == "EMG561"
    assert edf.recording.investigator_technician_code == "BK/JOP"
    assert edf.recording.equipment_code == "Sony."
    assert edf.recording.additional == ("MNC", "R", "Median", "Nerve.")


def test_edf_recording_setter(edf: Edf):
    edf.recording = Recording()
    assert edf.local_recording_identification == "Startdate X X X X"


def test_recording_raises_error_if_too_long_for_header_field():
    with pytest.raises(ValueError, match="exceeds maximum field length"):
        Recording(equipment_code="X" * 81)


def test_recording_from_str_raises_error_if_too_long_for_header_field():
    with pytest.raises(ValueError, match="exceeds maximum field length"):
        Recording._from_str("X" * 81)


def test_recording_raises_error_on_invalid_characters():
    with pytest.raises(UnicodeEncodeError, match="'ascii' codec can't encode"):
        Recording(investigator_technician_code="ÄÖÜ")


def test_recording_from_str_raises_error_on_invalid_characters():
    with pytest.raises(UnicodeEncodeError, match="'ascii' codec can't encode"):
        Recording._from_str("Startdate X ÄÖÜ X X")


def test_setting_edf_recording_with_new_startdate_changes_both_startdate_fields():
    new_date = datetime.date(2023, 4, 25)
    edf = Edf([EdfSignal(np.arange(10), 1)])
    edf.recording = Recording(startdate=new_date)
    assert edf.startdate == new_date
    assert edf.__startdate == b"25.04.23"
    assert edf.recording.startdate == new_date


def test_edf_startdate_raises_error_for_edf_instantiated_without_startdate():
    edf = Edf([EdfSignal(np.arange(10), 1)])
    with pytest.raises(AnonymizedDateError, match="startdate is not available"):
        edf.startdate
    with pytest.raises(AnonymizedDateError, match="startdate is not available"):
        edf.recording.startdate


def test_edf_startdate_raises_error_after_setting_anonymous_startdate():
    edf = Edf(
        [EdfSignal(np.arange(10), 1)],
        recording=Recording(startdate=datetime.date(2023, 4, 25)),
    )
    edf.recording = Recording(startdate=None)
    with pytest.raises(AnonymizedDateError, match="startdate is not available"):
        edf.startdate
    with pytest.raises(AnonymizedDateError, match="startdate is not available"):
        edf.recording.startdate


def test_setting_edf_startdate_changes_both_startdate_fields():
    new_date = datetime.date(2023, 4, 25)
    edf = Edf([EdfSignal(np.arange(10), 1)])
    edf.startdate = new_date
    assert edf.startdate == new_date
    assert edf.__startdate == b"25.04.23"
    assert edf.recording.startdate == new_date


@pytest.mark.parametrize(
    "local_recording_identification",
    LOCAL_RECORDING_IDENTIFICATION_WITH_INVALID_STARTDATE,
)
def test_setting_edf_startdate_does_not_modify_recording_field_if_startdate_subfield_is_invalid(
    local_recording_identification: bytes,
):
    edf = Edf([EdfSignal(np.arange(10), 1)])
    edf._local_recording_identification = local_recording_identification.ljust(80)
    edf.startdate = datetime.date(2001, 1, 1)
    assert edf.local_recording_identification == local_recording_identification.decode()


@pytest.mark.parametrize(
    "local_recording_identification",
    [
        b"Startdate 02-MAR-2002",
        b"Startdate 02-MAR-2002 X",
        b"Startdate 02-MAR-2002 hospital investigator equipmentcode",
        b"Startdate X X",
        b"Startdate X X X",
        b"Startdate X X X X",
        b"Startdate X X X X X",
    ],
)
def test_setting_edf_startdate_modifies_only_startdate_subfield_in_recording_field(
    local_recording_identification: bytes,
):
    edf = Edf([EdfSignal(np.arange(10), 1)])
    edf._local_recording_identification = local_recording_identification.ljust(80)
    edf.startdate = datetime.date(2001, 1, 1)
    subfields = local_recording_identification.split()
    subfields[1] = b"01-JAN-2001"
    assert edf.local_recording_identification == b" ".join(subfields).decode()


def test_edf_startdate_warns_on_differing_startdate_fields():
    edf = Edf(
        [EdfSignal(np.arange(10), 1)],
        recording=Recording(startdate=datetime.date(2023, 4, 25)),
    )
    edf.__startdate = b"01.01.01"
    with pytest.warns(UserWarning, match="Different values in startdate fields"):
        edf.startdate


def test_edf_startdate_raises_error_if_edfplus_startdate_subfield_is_anonymized():
    edf = Edf([EdfSignal(np.arange(10), 1)])
    with pytest.raises(ValueError, match="startdate is not available"):
        edf.startdate


@pytest.mark.parametrize(
    "local_recording_identification",
    LOCAL_RECORDING_IDENTIFICATION_WITH_INVALID_STARTDATE,
)
def test_edf_startdate_falls_back_to_legacy_field_if_recording_field_is_not_valid_edfplus(
    local_recording_identification: bytes,
):
    startdate = datetime.date(2001, 1, 1)
    edf = Edf([EdfSignal(np.arange(10), 1)], recording=Recording(startdate=startdate))
    edf._local_recording_identification = local_recording_identification.ljust(80)
    assert edf.startdate == startdate


@pytest.mark.parametrize(
    ("code", "sex", "name", "additional"),
    [
        ("", "X", "X", ()),
        ("X", "", "", ()),
        ("X", "X", "X", ("add1", "", "add2")),
        (None, "X", "X", None),
        ("X", None, None, None),
        ("X", "X", "X", (None, "add2")),
    ],
)
def test_patient_assumes_unspecified_subfields_as_unknown(
    code: str | None,
    sex: str | None,
    name: str | None,
    additional: tuple[str | None, ...] | None,
):
    patient = Patient(code=code, sex=sex, name=name, additional=additional)
    assert patient.code == code if code else "X"
    assert patient.sex == sex if sex else "X"
    assert patient.name == name if name else "X"
    expected_additional = list(additional) if additional else []
    for i in range(len(expected_additional)):
        if expected_additional[i] is None or expected_additional[i] == "":
            expected_additional[i] = "X"
    assert patient.additional == tuple(expected_additional)


def test_read_patient_all_subfields_missing():
    patient = Patient._from_str("")
    assert patient.code is None
    assert patient.sex is None
    assert patient.name is None
    assert patient.additional == ()
    with pytest.raises(AnonymizedDateError, match="birthdate is not available"):
        patient.birthdate


def test_read_patient_some_subfields_missing():
    patient = Patient._from_str("X M 21-AUG-1984")
    assert patient.code == "X"
    assert patient.sex == "M"
    assert patient.name is None
    assert patient.birthdate == datetime.date(1984, 8, 21)
    assert patient.additional == ()


@pytest.mark.parametrize(
    (
        "hospital_administration_code",
        "investigator_technician_code",
        "equipment_code",
        "additional",
    ),
    [
        ("X", "X", "X", ()),
        ("X", "X", "", ()),
        ("X", "", "X", ()),
        ("", "X", "X", ()),
        ("X", "", "", ()),
        ("", "X", "", ()),
        ("", "", "X", ()),
        ("", "", "", ()),
        ("X", "X", None, None),
        ("X", None, "X", None),
        (None, "X", "X", None),
        ("X", None, None, None),
        (None, "X", None, None),
        (None, None, "X", None),
        (None, None, None, None),
        ("X", "X", "X", ("add1", "add2")),
        ("X", "X", "X", ("add1", "")),
        ("X", "X", "X", (None, "add2")),
    ],
)
def test_recording_assumes_unspecified_subfields_as_unknown(
    hospital_administration_code: str | None,
    investigator_technician_code: str | None,
    equipment_code: str | None,
    additional: tuple[str | None, ...] | None,
):
    recording = Recording(
        startdate=datetime.date(2002, 3, 2),
        hospital_administration_code=hospital_administration_code,
        investigator_technician_code=investigator_technician_code,
        equipment_code=equipment_code,
        additional=additional,
    )
    assert (
        recording.hospital_administration_code == hospital_administration_code
        if hospital_administration_code
        else "X"
    )
    assert (
        recording.investigator_technician_code == investigator_technician_code
        if investigator_technician_code
        else "X"
    )
    assert recording.equipment_code == equipment_code if equipment_code else "X"
    expected_additional = list(additional) if additional else []
    for i in range(len(expected_additional)):
        if expected_additional[i] is None or expected_additional[i] == "":
            expected_additional[i] = "X"
    assert recording.additional == tuple(expected_additional)


def test_read_recording_all_subfields_missing():
    recording = Recording._from_str("Startdate")
    assert recording.hospital_administration_code is None
    assert recording.investigator_technician_code is None
    assert recording.equipment_code is None
    assert recording.additional == ()
    with pytest.raises(AnonymizedDateError, match="startdate is not available"):
        recording.startdate


def test_read_recording_some_subfields_missing():
    recording = Recording._from_str("Startdate 13-MAY-2025 X")
    assert recording.hospital_administration_code == "X"
    assert recording.investigator_technician_code is None
    assert recording.equipment_code is None
    assert recording.additional == ()
    assert recording.startdate == datetime.date(2025, 5, 13)

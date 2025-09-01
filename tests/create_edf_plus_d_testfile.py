#!/usr/bin/env python3
# produced by claude - not convinced it is correct as it is using edfio
# to write the time annotations
"""Script to create a short, de-identified EDF+D test file."""

import datetime
import numpy as np
from pathlib import Path

from edfio import Edf, EdfSignal, EdfAnnotation
from edfio.edf_header import Patient, Recording


def create_short_edf_plus_d():
    """Create a short EDF+D test file with de-identified data."""

    # Create de-identified patient info
    patient = Patient(
        code="X",
        sex="X",
        birthdate=None,  # None for anonymized birthdate
        name="X",
    )

    # Create recording info with a generic date
    recording = Recording(
        startdate=datetime.date(2020, 1, 1),
        hospital_administration_code="X",
        investigator_technician_code="X",
        equipment_code="X",
    )

    # Create a simple test signal (2 seconds at 100 Hz = 200 samples)
    # Use a simple sine wave pattern for predictable test data
    sampling_freq = 100
    duration = 2.0  # seconds
    num_samples = int(sampling_freq * duration)

    # Create a sine wave with some noise
    t = np.linspace(0, duration, num_samples)
    signal_data = 10 * np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.random.randn(num_samples)

    signal = EdfSignal(
        data=signal_data,
        sampling_frequency=sampling_freq,
        label="TestSignal",
        physical_dimension="uV",
        physical_range=(-100.0, 100.0),
    )

    # Create annotations with time stamps that will make this EDF+D
    # EDF+D requires annotations with relative time stamps
    annotations = [
        EdfAnnotation(0.0, None, "+0.0"),  # Time-stamp annotation for EDF+D
        EdfAnnotation(0.5, 0.1, "Event1"),
        EdfAnnotation(1.2, 0.1, "Event2"),
        EdfAnnotation(1.0, None, "+9.1"),  # Another time-stamp for discontinuity
    ]

    # Create the EDF file with 0.5 second data records
    edf = Edf(
        signals=[signal],
        annotations=annotations,
        patient=patient,
        recording=recording,
        starttime=datetime.time(12, 0, 0),
        data_record_duration=0.5,  # 0.5 second records
    )

    # Set as EDF+D explicitly
    edf._set_reserved("EDF+D")

    # Write to test data directory
    output_path = Path("tests/TEST_DATA/test_edf_plus_d.edf")
    edf.write(output_path)

    print(f"Created EDF+D test file: {output_path}")
    print(f"File size: {output_path.stat().st_size} bytes")

    # Verify it can be read back
    from edfio import read_edf

    test_edf = read_edf(output_path)
    print(f"Format: {test_edf.edf_format}")
    print(f"Is discontinuous: {test_edf.is_discontinuous}")
    print(f"Number of data records: {test_edf.num_data_records}")
    print(f"Data record duration: {test_edf.data_record_duration}")
    print(f"Number of annotations: {len(test_edf.annotations)}")


if __name__ == "__main__":
    create_short_edf_plus_d()

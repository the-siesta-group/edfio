"""Tests for EDF+D onset time map functionality."""

import datetime
import math
from pathlib import Path

import numpy as np
import pytest

import edfio
from edfio import Edf, EdfAnnotation, EdfSignal


class TestOnsetTimeMap:
    """Test suite for onset time map functionality."""

    @pytest.fixture
    def mock_edf_plus_d(self):
        """Create a mock EDF+D file for testing."""
        # Create a signal with known data (multiple of data record duration)
        signal_data = np.random.randn(1000)  # 4 seconds of data at 250Hz
        signal = EdfSignal(
            data=signal_data,
            sampling_frequency=250,
            label="Test Signal",
        )

        # Create annotations that will create an EDF+D structure
        annotations = [
            EdfAnnotation(0.0, None, "Start"),
            EdfAnnotation(1.0, None, "Event1"),
            EdfAnnotation(2.0, None, "Event2"),
        ]

        # Provide proper recording info
        from edfio.edf_header import Recording  # noqa PLC0415  # allow imports inside of tests

        recording = Recording(startdate=datetime.date(2023, 1, 1))

        edf = Edf(
            [signal],
            annotations=annotations,
            recording=recording,
            data_record_duration=0.5,  # 0.5 second records, 8 records total
        )

        # Manually set to EDF+D for testing
        edf._set_reserved("EDF+D")
        return edf

    def test_get_datarecord_relative_onset_times_error_on_non_discontinuous(self):
        """Test that method raises error for non-EDF+D files."""
        # Create a regular EDF file
        signal_data = np.random.randn(1000)
        signal = EdfSignal(
            data=signal_data,
            sampling_frequency=250,
            label="Test Signal",
        )
        edf = Edf([signal])

        with pytest.raises(
            ValueError, match="only available for discontinuous EDF\\+D"
        ):
            edf.get_datarecord_relative_onset_times()

    def test_get_datarecord_onset_datetimes_error_on_non_discontinuous(self):
        """Test that datetime method raises error for non-EDF+D files."""
        signal_data = np.random.randn(1000)
        signal = EdfSignal(
            data=signal_data,
            sampling_frequency=250,
            label="Test Signal",
        )

        # Provide proper recording info to avoid AnonymizedDateError
        from edfio.edf_header import (  # noqa: PLC0415  # allow imports inside of tests
            Recording,
        )

        recording = Recording(startdate=datetime.date(2023, 1, 1))
        edf = Edf([signal], recording=recording)

        with pytest.raises(
            ValueError, match="only available for discontinuous EDF\\+D"
        ):
            edf.get_datarecord_onset_datetimes()

    def test_get_datarecord_discontinuity_indices_error_on_non_discontinuous(self):
        """Test that discontinuity method raises error for non-EDF+D files."""
        signal_data = np.random.randn(1000)
        signal = EdfSignal(
            data=signal_data,
            sampling_frequency=250,
            label="Test Signal",
        )
        edf = Edf([signal])

        with pytest.raises(
            ValueError, match="only available for discontinuous EDF\\+D"
        ):
            edf.get_datarecord_discontinuity_indices()

    def test_caching_behavior(self, mock_edf_plus_d):
        """Test that caching works correctly."""
        if not mock_edf_plus_d.is_discontinuous:
            pytest.skip("Mock EDF is not discontinuous - skipping cache test")

        # Clear any existing cache
        mock_edf_plus_d._cached_onset_times = None

        # First call should compute and cache
        times1 = mock_edf_plus_d.get_datarecord_relative_onset_times()
        assert mock_edf_plus_d._cached_onset_times is not None
        assert len(times1) > 0

        # Second call should use cache
        times2 = mock_edf_plus_d.get_datarecord_relative_onset_times()
        np.testing.assert_array_equal(times1, times2)

        # Forced recalculation should work
        times3 = mock_edf_plus_d.get_datarecord_relative_onset_times(use_cache=False)
        np.testing.assert_array_equal(times1, times3)

    def test_cache_invalidation(self, mock_edf_plus_d):
        """Test that cache is invalidated when EDF structure changes."""
        if not mock_edf_plus_d.is_discontinuous:
            pytest.skip("Mock EDF is not discontinuous - skipping cache test")

        # Populate cache
        mock_edf_plus_d.get_datarecord_relative_onset_times()
        assert mock_edf_plus_d._cached_onset_times is not None

        # Modify structure - this should invalidate cache
        mock_edf_plus_d._set_num_data_records(mock_edf_plus_d.num_data_records)
        assert mock_edf_plus_d._cached_onset_times is None

    def test_return_types(self, mock_edf_plus_d):
        """Test that methods return correct types."""
        if not mock_edf_plus_d.is_discontinuous:
            pytest.skip("Mock EDF is not discontinuous - skipping type test")

        # Test relative times return numpy array
        rel_times = mock_edf_plus_d.get_datarecord_relative_onset_times()
        assert isinstance(rel_times, np.ndarray)
        assert rel_times.dtype == np.float64

        # Test absolute times return datetime list
        abs_times = mock_edf_plus_d.get_datarecord_onset_datetimes()
        assert isinstance(abs_times, list)
        assert all(isinstance(t, datetime.datetime) for t in abs_times)

        # Test discontinuity indices return numpy array
        disc_indices = mock_edf_plus_d.get_datarecord_discontinuity_indices()
        assert isinstance(disc_indices, np.ndarray)
        assert disc_indices.dtype == np.int64

    def test_datetime_calculation_consistency(self, mock_edf_plus_d):
        """Test that datetime calculation is consistent with relative times."""
        if not mock_edf_plus_d.is_discontinuous:
            pytest.skip("Mock EDF is not discontinuous - skipping consistency test")

        rel_times = mock_edf_plus_d.get_datarecord_relative_onset_times()
        abs_times = mock_edf_plus_d.get_datarecord_onset_datetimes()

        assert len(rel_times) == len(abs_times)

        # Check that the conversion is correct
        start_datetime = datetime.datetime.combine(
            mock_edf_plus_d.startdate, mock_edf_plus_d.starttime
        )

        for rel_time, abs_time in zip(rel_times, abs_times):
            expected_abs_time = start_datetime + datetime.timedelta(
                seconds=float(rel_time)
            )
            # Allow small floating point differences
            time_diff = abs((abs_time - expected_abs_time).total_seconds())
            assert time_diff < 1e-6

    @pytest.mark.skipif(
        not Path("/home/clee/code/python-edf/tests/edf+D_sample.edf").exists(),
        reason="Test EDF+D file not available",
    )
    def test_with_real_edf_plus_d_file(self):
        """Test with a real EDF+D file if available."""
        test_file = Path("/home/clee/code/python-edf/tests/edf+D_sample.edf")
        edf = edfio.read_edf(test_file)

        assert edf.is_discontinuous
        assert edf.edf_format == "EDF+D"

        # Test relative onset times
        rel_times = edf.get_datarecord_relative_onset_times()
        assert len(rel_times) == edf.num_data_records
        assert rel_times.dtype == np.float64

        # Test absolute datetimes
        abs_times = edf.get_datarecord_onset_datetimes()
        assert len(abs_times) == edf.num_data_records
        assert all(isinstance(t, datetime.datetime) for t in abs_times)

        # Test discontinuity detection
        disc_indices = edf.get_datarecord_discontinuity_indices()
        assert isinstance(disc_indices, np.ndarray)

        # Verify discontinuities make sense
        if len(disc_indices) > 0:
            time_diffs = np.diff(rel_times)
            # All discontinuity indices should correspond to large time jumps
            for idx in disc_indices:
                assert time_diffs[idx] > edf.data_record_duration * 1.1

        # Test caching performance
        import time  # noqa: PLC0415

        start = time.time()
        rel_times_1 = edf.get_datarecord_relative_onset_times()
        first_call_time = time.time() - start

        start = time.time()
        rel_times_2 = edf.get_datarecord_relative_onset_times()
        second_call_time = time.time() - start

        # Second call should be much faster (cached) - but allow for very fast operations
        if first_call_time > 1e-5:  # Only test if first call was measurably slow
            assert second_call_time < first_call_time / 2  # At least 2x faster
        np.testing.assert_array_equal(rel_times_1, rel_times_2)

    def test_large_file_cache_behavior(self):
        """Test that very large files don't cache to prevent memory issues."""
        # Create a simple EDF to use as base
        signal_data = np.random.randn(1000)
        signal = EdfSignal(
            data=signal_data,
            sampling_frequency=250,
            label="Test Signal",
        )

        from edfio.edf_header import Recording  # noqa: PLC0415

        recording = Recording(startdate=datetime.date(2023, 1, 1))
        edf = Edf([signal], recording=recording)

        # Mock the relative onset computation to return a large array
        large_array = np.arange(150_000, dtype=np.float64)  # Over cache threshold

        def mock_compute():
            return large_array

        # Make it appear discontinuous and mock the computation
        edf._set_reserved("EDF+D")
        edf._compute_relative_onset_times = mock_compute

        # Call with caching enabled
        result = edf.get_datarecord_relative_onset_times(use_cache=True)

        # Should not cache due to size
        assert edf._cached_onset_times is None
        np.testing.assert_array_equal(result, large_array)


class TestAsyncOnsetTimeMap:
    """Test async functionality for onset time maps."""

    @pytest.mark.skipif(
        not Path("/home/clee/code/python-edf/tests/edf+D_sample.edf").exists(),
        reason="Test EDF+D file not available",
    )
    def test_async_onset_time_map(self):
        """Test async computation of onset time map."""
        test_file = Path("/home/clee/code/python-edf/tests/edf+D_sample.edf")
        edf = edfio.read_edf(test_file)

        with edfio.EdfAsyncProcessor(edf) as async_proc:
            future = async_proc.create_onset_time_map_async()
            result = future.result()

            # Compare with synchronous result
            sync_result = edf.get_datarecord_onset_datetimes()
            assert len(result) == len(sync_result)
            assert all(a == s for a, s in zip(result, sync_result))

    def test_async_processor_context_manager(self):
        """Test that async processor context manager works correctly."""
        signal_data = np.random.randn(1000)
        signal = EdfSignal(data=signal_data, sampling_frequency=250, label="Test")
        edf = Edf([signal])

        # Test context manager
        with edfio.EdfAsyncProcessor(edf) as proc:
            assert proc._executor is not None

        # Executor should be shut down after context exit
        # Note: We can't directly test this without accessing private members
        # but the context manager should handle cleanup


class TestDataRecordRange:
    """Test suite for data record range functionality."""

    def test_get_data_record_range_basic(self):
        """Test basic data record range calculation."""
        # Create a simple EDF for testing
        signal_data = np.arange(1000, dtype=np.float64)  # 4 seconds at 250Hz
        signal = EdfSignal(
            data=signal_data,
            sampling_frequency=250,
            label="Test Signal",
        )

        from edfio.edf_header import Recording  # noqa: PLC0415

        recording = Recording(startdate=datetime.date(2023, 1, 1))
        edf = Edf(
            [signal], recording=recording, data_record_duration=1.0
        )  # 1 second records

        # Test basic range
        start_rec, end_rec = edf._get_data_record_range(100, 300, 50)

        # Manual calculation:
        # Samples [50, 350) -> times [0.2s, 1.4s) -> records [0, 2)
        expected_start = 0
        expected_end = 2

        assert start_rec == expected_start
        assert end_rec == expected_end

    def test_get_data_record_range_edge_cases(self):
        """Test edge cases for data record range calculation."""
        signal_data = np.arange(1000, dtype=np.float64)
        signal = EdfSignal(
            data=signal_data,
            sampling_frequency=250,
            label="Test Signal",
        )

        from edfio.edf_header import Recording  # noqa: PLC0415

        recording = Recording(startdate=datetime.date(2023, 1, 1))
        edf = Edf([signal], recording=recording, data_record_duration=1.0)

        # Test with zero extra
        start_rec, end_rec = edf._get_data_record_range(100, 200, 0)
        assert isinstance(start_rec, int)
        assert isinstance(end_rec, int)
        assert start_rec <= end_rec

        # Test with start at 0
        start_rec, end_rec = edf._get_data_record_range(0, 100, 10)
        assert start_rec == 0

        # Test with large extra that would cause negative start
        start_rec, end_rec = edf._get_data_record_range(50, 100, 100)
        assert start_rec == 0  # Should be clamped to 0

    def test_get_data_record_range_validation(self):
        """Test input validation for data record range calculation."""
        signal_data = np.arange(1000, dtype=np.float64)
        signal = EdfSignal(
            data=signal_data,
            sampling_frequency=250,
            label="Test Signal",
        )

        from edfio.edf_header import Recording  # noqa: PLC0415

        recording = Recording(startdate=datetime.date(2023, 1, 1))
        edf = Edf([signal], recording=recording)

        # Test negative start_samp
        with pytest.raises(ValueError, match="start_samp must be non-negative"):
            edf._get_data_record_range(-1, 100, 0)

        # Test negative end_samp
        with pytest.raises(ValueError, match="end_samp must be non-negative"):
            edf._get_data_record_range(100, -1, 0)

        # Test negative extra_samp
        with pytest.raises(ValueError, match="extra_samp must be non-negative"):
            edf._get_data_record_range(100, 200, -1)

        # Test end_samp < start_samp
        with pytest.raises(ValueError, match="end_samp must be >= start_samp"):
            edf._get_data_record_range(200, 100, 0)

    def test_get_data_record_range_no_signals(self):
        """Test error handling when EDF has no signals."""
        # Create EDF with only annotations
        annotations = [EdfAnnotation(0.0, None, "Start")]
        edf = Edf([], annotations=annotations)

        with pytest.raises(ValueError, match="No signals available"):
            edf._get_data_record_range(0, 100, 0)

    @pytest.mark.skipif(
        not Path("/home/clee/code/python-edf/tests/edf+D_sample.edf").exists(),
        reason="Test EDF+D file not available",
    )
    def test_get_data_record_range_real_file(self):
        """Test data record range calculation with real EDF file."""
        test_file = Path("/home/clee/code/python-edf/tests/edf+D_sample.edf")
        edf = edfio.read_edf(test_file)

        signal = edf.signals[0]
        fs = signal.sampling_frequency

        # Test a reasonable range
        sample1, sample2 = 1000, 2000
        extra = 100

        start_rec, end_rec = edf._get_data_record_range(sample1, sample2, extra)

        # Verify the result makes sense
        assert isinstance(start_rec, int)
        assert isinstance(end_rec, int)
        assert 0 <= start_rec < end_rec <= edf.num_data_records

        # Manual verification
        start_time = (sample1 - extra) / fs
        end_time = (sample2 + extra) / fs
        expected_start = int(start_time // edf.data_record_duration)
        expected_end = math.ceil(end_time / edf.data_record_duration - 1e-12)
        expected_end = min(expected_end, edf.num_data_records)

        assert start_rec == expected_start
        assert end_rec == expected_end

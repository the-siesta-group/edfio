"""Asynchronous utilities for EDF file processing."""

from __future__ import annotations

import datetime
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edfio.edf import Edf


class EdfAsyncProcessor:
    """Asynchronous processor for EDF operations."""

    def __init__(self, edf: Edf) -> None:
        """
        Initialize the async processor for an EDF object.

        Parameters
        ----------
        edf : Edf
            The EDF object to process.
        """
        self._edf = edf
        self._executor = ThreadPoolExecutor(max_workers=1)

    def create_onset_time_map_async(self) -> Future[list[datetime.datetime]]:
        """
        Create onset time map asynchronously in a background thread.

        This method computes the absolute datetime for each data record onset
        in EDF+D files, running the computation in a separate thread to avoid
        blocking the main thread.

        Returns
        -------
        Future[list[datetime.datetime]]
            A Future object that will contain the list of datetime objects
            representing the absolute time of each data record onset.

        Raises
        ------
        ValueError
            If the EDF file is not an EDF+D discontinuous file.

        Examples
        --------
        >>> import edfio
        >>> edf = edfio.read_edf("discontinuous.edf")
        >>> async_processor = EdfAsyncProcessor(edf)
        >>> future = async_processor.create_onset_time_map_async()
        >>> # Do other work while computation runs...
        >>> time_map = future.result()  # Get the result when ready
        """
        return self._executor.submit(self._edf.get_datarecord_onset_datetimes)

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=True)

    def __enter__(self) -> EdfAsyncProcessor:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: type) -> None:
        """Context manager exit."""
        self.shutdown()

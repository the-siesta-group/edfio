from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray


class LazyLoader:
    """
    Class to load data for a single signal from a buffer of EDF data records (array or memmap).

    Parameters
    ----------
    buffer : numpy.ndarray or numpy.memmap
        Buffer of EDF data records.
    start_sample : int
        Offset of the signal in the data records (in samples).
    end_sample : int
        End of the signal in the data records (in samples).
    """

    def __init__(
        self,
        buffer: Union[NDArray[np.int16], np.memmap[Any, np.dtype[np.int16]]],
        start_sample: int,
        end_sample: int,
    ) -> None:
        self.buffer = buffer
        self.start_sample = start_sample
        self.end_sample = end_sample

    def load(
        self, start_record: Optional[int] = None, end_record: Optional[int] = None
    ) -> NDArray[np.int16]:
        """
        Load signal data from the buffer.

        Parameters
        ----------
        start_record : int, optional
            The first EDF data record to load samples from. If None, load from the beginning of the buffer.
        end_record : int, optional
            The last EDF data record to load samples from. If None, load until the end of the buffer.

        Returns
        -------
        numpy.ndarray
            Signal data (digital).
        """
        if start_record is None:
            start_record = 0
        if end_record is None:
            end_record = self.buffer.shape[0]
        if (
            end_record < start_record
            or start_record < 0
            or end_record > self.buffer.shape[0]
        ):
            raise ValueError("Invalid slice: Slice exceeds EDF duration")
        return self.buffer[
            start_record:end_record, self.start_sample : self.end_sample
        ].flatten()

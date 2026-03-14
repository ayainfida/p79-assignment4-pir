import numpy as np
from .message import PIRScheme

"""
This module defines a very simple Database class, which will be used in the PIR scheme.
It supports standard get and set operations, and it can be initialized with a specific PIR scheme: NAIVE, SQRT, or OPTIMIZED_SQRT.
"""
class Database:
    def __init__(self, N: int, data: list = None, scheme: PIRScheme = PIRScheme.SQRT, dtype=bool):
        assert N > 0, "N must be a positive integer."
        # For simiplicity, I will assume that N is complete square.
        assert np.sqrt(N) == int(np.sqrt(N)), "N must be a perfect square."
        assert dtype in (bool, np.uint8), "Invalid datatype. Supported datatypes are bool and uint8."

        self.N = N
        self.scheme = scheme

        if data is not None:
            assert len(data) == N, f"Data length must be equal to N. Given length: {len(data)} and N: {N}"
            self.data = np.array(data, dtype=dtype)
        else:
            self.data = np.zeros(N, dtype=dtype)

        # Incase of naive scheme, we can represent the database as a 1D array of booleans.
        # For the sqrt scheme, we can represent the database as a 2D array of booleans with dimensions sqrt(N) x sqrt(N).
        # Numpy arrays are used due to efficient access and its support for vectorized operations.
        if scheme == PIRScheme.SQRT or scheme == PIRScheme.OPTIMIZED_SQRT:
            self.data = self.data.reshape((int(np.sqrt(N)), int(np.sqrt(N))))
        elif scheme != PIRScheme.NAIVE:
            raise ValueError(f"Invalid PIR scheme. Given: {scheme}")
        
        # Logs in case of updates to the database, which will then be used in the OPTIMIZED_SQRT scheme to update hints for the client.
        self.update_log = []

    def get_dimensions(self) -> tuple[int, int]:
        return self.data.shape
    
    def object(self) -> np.ndarray[int]:
        """
        Returns a numpy object of database.
        """
        bit_matrix = np.stack([(self.data >> k) & 1 for k in range(8 if self.data.dtype == np.uint8 else 1)], axis=0)
        return bit_matrix
    
    def get_row_col(self, idx: int) -> tuple[int, int]:
        """
        Get the row and column indices for the given index in the database.
        """
        row = idx // self.get_dimensions()[1]
        col = idx % self.get_dimensions()[1]
        return row, col
    
    def get(self, idx: int) -> int:
        """
        Get the value at the given index in the database.
        """
        assert 0 <= idx < self.N, f"Index out of bounds. Given index: {idx} and N: {self.N}"

        if self.scheme == PIRScheme.NAIVE:
            return int(self.data[idx])
        elif self.scheme == PIRScheme.SQRT or self.scheme == PIRScheme.OPTIMIZED_SQRT:
            row, col = self.get_row_col(idx)
            return int(self.data[row, col])
        
    def set(self, idx: int, value: int) -> None:
        """
        Set the value at the given index in the database.
        """
        assert 0 <= idx < self.N, f"Index out of bounds. Given index: {idx} and N: {self.N}"
        assert value in (0, 1) if self.data.dtype == bool else True, f"Value must be 0 or 1 for boolean databases. Given value: {value}"
        assert 0 <= value <= 255 if self.data.dtype == np.uint8 else True, f"Value must be between 0 and 255 for uint8 databases. Given value: {value}"

        if self.scheme == PIRScheme.NAIVE:
            self.data[idx] = value
        elif self.scheme == PIRScheme.SQRT:
            row, col = self.get_row_col(idx)
            self.data[row, col] = value
        elif self.scheme == PIRScheme.OPTIMIZED_SQRT:
            row, col = self.get_row_col(idx)
            # Log the update incase it's different from the current value
            if self.data[row, col] != value:
                self.update_log.append((idx, int(self.data[row, col]), value))
            self.data[row, col] = value
    
    def get_logs(self) -> list[tuple[int, int]]:
        """
        Get the update logs for the databas. 
        Avaliable only for the OPTIMIZED_SQRT scheme.
        """
        assert self.scheme == PIRScheme.OPTIMIZED_SQRT, f"Update logs are only available for the OPTIMIZED_SQRT scheme. Given scheme: {self.scheme}"
        return self.update_log

    def clear_logs(self) -> None:
        """
        Clear the update logs for the database.
        Avaliable only for the OPTIMIZED_SQRT scheme.
        """
        assert self.scheme == PIRScheme.OPTIMIZED_SQRT, f"Update logs are only available for the OPTIMIZED_SQRT scheme. Given scheme: {self.scheme}"
        self.update_log = []

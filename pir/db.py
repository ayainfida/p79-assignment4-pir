from .message import PIRScheme
import numpy as np

"""
This module defines a very simple Database class, which will be used in the PIR scheme.
It supports standard get and set operations, and it can be initialized with a specific PIR scheme: NAIVE, SQRT, or OPTIMIZED_SQRT.
"""
class Database:
    def __init__(self, N: int, data: list = None, scheme: PIRScheme = PIRScheme.SQRT, dtype=bool):
        assert N > 0, "N must be a positive integer."
        # For simiplicity, I will assume that N is complete square.
        assert np.sqrt(N) == int(np.sqrt(N)), "N must be a perfect square."

        self.N = N
        self.scheme = scheme

        if data is not None:
            assert len(data) == N, "Data length must be equal to N. Given length: {} and N: {}".format(len(data), N)
            self.data = np.array(data, dtype=dtype)
        else:
            self.data = np.zeros(N, dtype=dtype)

        # Incase of naive scheme, we can represent the database as a 1D array of booleans.
        # For the sqrt scheme, we can represent the database as a 2D array of booleans with dimensions sqrt(N) x sqrt(N).
        # Numpy arrays are used due to efficient access and its support for vectorized operations.
        if scheme == PIRScheme.SQRT or scheme == PIRScheme.OPTIMIZED_SQRT:
            self.data = self.data.reshape((int(np.sqrt(N)), int(np.sqrt(N))))
        elif scheme != PIRScheme.NAIVE:
            raise ValueError("Invalid PIR scheme. Given: {}".format(scheme))

    def get_dimensions(self) -> tuple[int, int]:
        return self.data.shape
    
    def object(self) -> np.ndarray[int]:
        """
        Returns a numpy object of database.
        """
        return self.data.astype(int)
    
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
        elif self.scheme == PIRScheme.SQRT or self.scheme == PIRScheme.OPTIMIZED_SQRT:
            row, col = self.get_row_col(idx)
            self.data[row, col] = value

if __name__ == "__main__":
    db = Database(16, [0]*16, PIRScheme.NAIVE, dtype=np.uint8)
    db.set(5, 1)
    print(db.data)
    db.set(15, 256)
    print(db.data)
    print(db.get(5))
    print(db.get(15))
    print(db.get(0))
    print(db.object())
    print(db.get_dimensions())
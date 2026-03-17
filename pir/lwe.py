import numpy as np
from pir.ring import RingElement

"""
This module contains the implementation of the LWE methods 
for sampling error vectors, secret vectors and generating the vector A.
These methods will be used by the PIR scheme.
"""
class LWEMethods:
    @staticmethod
    def sample_error_vector(N: int, q: int, dtype: type, B: int = 3) -> np.ndarray:
        """
        Sample small error vector from the interval [-B, B].

        Args:
            N: length of vector
            q: modulo integer for the ring
            dtype: data type for the vector, either np.uint8 or bool
            B: bound for error

        Returns:
            error vector of length N of RingElements
        """
        n_bits = 8 if dtype == np.uint8 else 1
        error = np.random.randint(-B, B + 1, size=N*n_bits)

        return RingElement.get_ring_vector(error, q).reshape(n_bits, N)

    @staticmethod
    def sample_secret_vector(N: int, q: int, dtype: type) -> np.ndarray:
        """
        Sample secret vector from over the interval {-q, q}.

        Args:
            N: length of vector
            q: modulo integer for the ring
            dtype: data type for the vector, either np.uint8 or bool

        Returns:
            secret vector of length N of RingElements
        """
        n_bits = 8 if dtype == np.uint8 else 1
        secret = np.random.randint(-q, q, size=N*n_bits)

        return RingElement.get_ring_vector(secret, q).reshape(n_bits, N)
    
    @staticmethod
    def generate_matrix_A(N: int, n: int, q: int, dtype: type, seed: int) -> np.ndarray:
        """
        Generate a random matrix A of size N x n with entries uniformly sampled from the ring given a fixed seed.

        Args:
            N: number of rows
            n: number of columns
            q: modulo integer for the ring
            dtype: data type for the matrix, either np.uint8 or bool
            seed: random seed

        Returns:
            A random matrix of size N x n of RingElements
        """
        np.random.seed(seed)
        n_bits = 8 if dtype == np.uint8 else 1
        A = np.random.randint(-q, q, size=(n_bits, N, n))
        
        return RingElement.get_ring_vector(A.flatten(), q).reshape(n_bits, N, n)
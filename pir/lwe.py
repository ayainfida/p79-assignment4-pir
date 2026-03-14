import numpy as np

from pir.ring import RingElement

# default parameters for LWE
q = 15 # modulo integer for the ring
n = 512 # security parameter
N = 1024 # number of samples, should be at least n*log(q) for security

"""
This module contains the implementation of the LWE methods for sampling error vectors, secret vectors and generating the vector A.
These methods will be used by the PIR scheme.
"""
class LWEMethods:
    @staticmethod
    def sample_error_vector(N: int, q: int, dtype: type, B: int = 3) -> np.ndarray[RingElement]:
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
    def sample_secret_vector(N: int, q: int, dtype: type) -> np.ndarray[RingElement]:
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
    def generate_matrix_A(N: int, n: int, q: int, dtype: type, seed: int) -> np.ndarray[RingElement]:
        """
        Generate a random matrix A of size M x N with entries uniformly sampled from the ring given a fixed seed.

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
    
if __name__ == "__main__":
    # Example usage of the LWE methods
    error_vector = LWEMethods.sample_error_vector(N=10, q =32, dtype=np.uint8)
    print("Sampled error vector:", error_vector)
    # print("Type of error vector elements:", type(error_vector[0]))
    # secret_vector = LWEMethods.sample_secret_vector(N=10)
    # # print("Sampled secret vector:", secret_vector)

    # A = LWEMethods.generate_matrix_A(N=10, n=10, q=15, seed=42)
    # # print("Generated matrix A:", A)
    # # print("Type of A matrix elements:", type(A[0, 0]))
    # from .encoding import encode_std_pir_query, decode_std_pir_query

    # # Example encoding and decoding of a standard PIR query
    # c = np.random.randint(-q, q, size=10)
    # print("Original c vector:", c)
    # c_ring = RingElement.get_ring_vector(c, q)
    # query_bytes = encode_std_pir_query(A, c_ring)
    # print("Encoded query bytes length:", len(query_bytes))
    # A_decoded, c_decoded = decode_std_pir_query(query_bytes, N=10, n=10)
    # print("Decoded A matrix:", np.array_equal(A_decoded, A))
    # print("Decoded c vector:", np.array_equal(c_decoded, c_ring))
    # print(RingElement.extract_normal_vector(c_ring))
    # print(RingElement.extract_normal_vector(c_decoded))
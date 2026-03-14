import numpy as np
from .ring import RingElement


def decode_little_endian(b: bytes) -> int:
    """
    Decode a little-endian byte sequence to an integer.
    """
    return sum((b[i] << (8 * i)) for i in range(len(b)))

def encode_little_endian(x: int, length: int = 32) -> bytes:
    """
    Encode an integer to a little-endian byte sequence of the specified length.
    """
    b = bytearray(length)
    for i in range(length):
        b[i] = (x >> (8 * i)) & 0xFF
    return bytes(b)

def encode_std_pir(A: np.ndarray[RingElement], c: np.ndarray[RingElement], length: int = 32) -> bytes:
    """
    Encode the standard PIR query into bytes:
    A | c
    """
    # Encode A and c as bytes
    A_bytes = b''.join(encode_little_endian(a.value, length) for a in A.flatten())
    c_bytes = b''.join(encode_little_endian(c_i.value, length) for c_i in c.flatten())

    return A_bytes + c_bytes  

def decode_std_pir(data: bytes, N: int, n: int, q: int, dtype: type, length: int = 32) -> tuple[np.ndarray[RingElement], np.ndarray[RingElement]]:
    """
    Decode the standard PIR query from bytes:
    A | c
    """
    # We first calculate the expected length of A and c in bytes
    n_bits = 8 if dtype == np.uint8 else 1
    A_length = N * n * length * n_bits
    c_length = N * length  * n_bits 

    assert len(data) == A_length + c_length, f"Invalid data length for decoding. Expected: {A_length + c_length} and got: {len(data)}"

    # Decode A and c from bytes
    A = RingElement.get_ring_vector(np.array([decode_little_endian(data[i:i+length]) for i in range(0, A_length, length)], dtype=object), q).reshape(n_bits, N, n)
    c = RingElement.get_ring_vector(np.array([decode_little_endian(data[A_length + i:A_length + i + length]) for i in range(0, c_length, length)], dtype=object), q).reshape(n_bits, N)

    return A, c

def encode_opt_pir(c: np.ndarray[RingElement], length: int = 32) -> bytes:
    """
    Encode the vector c into bytes
    """
    return b''.join(encode_little_endian(c_i.value, length) for c_i in c.flatten())

def decode_opt_pir(data: bytes, N: int, q: int, dtype: type, length: int = 32) -> np.ndarray[RingElement]:
    """
    Decode the vector c from bytes
    """
    n_bits = 8 if dtype == np.uint8 else 1
    assert len(data) == N * length * n_bits, f"Invalid data length for decoding. Expected: {N * length * n_bits} and got: {len(data)}"

    c = RingElement.get_ring_vector(np.array([decode_little_endian(data[i:i+length]) for i in range(0, len(data), length)], dtype=object), q).reshape(n_bits, N)

    return c

def encode_hint(seed: int, A: np.ndarray[RingElement], length: int = 32) -> bytes:
    """
    Encode the hint into bytes.
    """
    A_bytes = b''.join(encode_little_endian(a.value, length) for a in A.flatten())
    seed_bytes = encode_little_endian(seed, length)

    return seed_bytes + A_bytes

def decode_hint(data: bytes, N: int, n: int, q: int, dtype: type, length: int = 32) -> tuple[int, np.ndarray[RingElement]]:
    """
    Decode the hint from bytes.
    """
    n_bits = 8 if dtype == np.uint8 else 1
    A_length = N * n * length * n_bits

    assert len(data) == length + A_length, f"Invalid data length for decoding. Expected: {length + A_length} and got: {len(data)}"
    seed = decode_little_endian(data[:length])
    A = RingElement.get_ring_vector(np.array([decode_little_endian(data[length + i:length + i + length]) for i in range(0, A_length, length)], dtype=object), q)
    
    A = A.reshape(n_bits, N, n)

    return seed, A
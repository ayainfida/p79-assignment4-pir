import random
import unittest
import numpy as np
from pir import PIRClient, PIRServer, PIRScheme, Database, PIRMessage
from pir.lwe import LWEMethods
from pir.encoding import decode_opt_pir, encode_opt_pir, encode_std_pir, decode_std_pir, encode_hint, decode_hint
from pir.ring import RingElement

class TestEncoding(unittest.TestCase):
    def setUp(self):
        # Let's set up some parameters for our tests
        self.A = LWEMethods.generate_matrix_A(N=4, n=4, q=16, dtype=np.uint8, seed=42)
        self.s = LWEMethods.sample_secret_vector(N=4, q=16, dtype=np.uint8)
        self.error = LWEMethods.sample_error_vector(N=4, q=16, dtype=np.uint8)
        self.seed = random.randint(0, 1000)

        # Let's assume C = A @ s + e (mod q) for the standard PIR query, we're not adding the query vector here for simplicity here
        self.c = ((self.A @ self.s[..., None]).squeeze(-1) + self.error)

    def test_encode_decode_std_pir(self):
        # Encode the standard PIR query
        encoded_query = encode_std_pir(self.A, self.c)

        # Decode the standard PIR query
        decoded_A, decoded_c = decode_std_pir(encoded_query, N=4, n=4, q=16, dtype=np.uint8)

        # 1) Check if the the decoded A and c have the same shape and values as the original A and c
        self.assertEqual(decoded_A.shape, self.A.shape)
        self.assertEqual(decoded_c.shape, self.c.shape)
        np.testing.assert_array_equal(decoded_A.flatten(), self.A.flatten())
        np.testing.assert_array_equal(decoded_c.flatten(), self.c.flatten())

        # 2) Encoding with invalid types should raise a AssertionError
        with self.assertRaises(AssertionError):
            encode_std_pir(np.zeros_like(self.A), self.c)
        with self.assertRaises(AssertionError):
            encode_std_pir(self.A, np.zeros_like(self.c))

        # 3) Decode with wrong length of data
        with self.assertRaises(AssertionError):
            decode_std_pir(encoded_query[:-1], N=4, n=4, q=16, dtype=np.uint8)
        with self.assertRaises(AssertionError):
            decode_std_pir(encoded_query + b'\x00', N=4, n=4, q=16, dtype=np.uint8)

        # 4) Decoding empty data should raise an assertion error
        with self.assertRaises(AssertionError):
            decode_std_pir(b'', N=4, n=4, q=16, dtype=np.uint8)
    
    def test_encode_decode_opt_pir(self):        
        # Encode the optimized PIR query (just the vector c)
        encoded_query = encode_opt_pir(self.c)

        # Decode the optimized PIR query
        decoded_c = decode_opt_pir(encoded_query, N=4, q=16, dtype=np.uint8)

        # 1) Check if the decoded c has the same shape and values as the original c
        self.assertEqual(decoded_c.shape, self.c.shape)
        np.testing.assert_array_equal(decoded_c.flatten(), self.c.flatten())
        
        # 2) Encoding with invalid types should raise a AssertionError
        with self.assertRaises(AssertionError):
            encode_opt_pir(np.zeros_like(self.c))

        # 3) Decode with wrong length of data
        with self.assertRaises(AssertionError):
            decode_opt_pir(encoded_query[:-1], N=4, q=16, dtype=np.uint8)
        with self.assertRaises(AssertionError):
            decode_opt_pir(encoded_query + b'\x00', N=4, q=16, dtype=np.uint8)
        
        # 4) Decoding empty data should raise an assertion error
        with self.assertRaises(AssertionError):
            decode_opt_pir(b'', N=4, q=16, dtype=np.uint8)

    def test_encode_decode_hint(self):
        # Encode the hint
        encoded_hint = encode_hint(self.seed, self.A)

        # Decode the hint
        decoded_seed, decoded_A = decode_hint(encoded_hint, N=4, n=4, q=16, dtype=np.uint8)

        # 1) Check if the decoded seed and A have the same values as the original seed and A
        self.assertEqual(decoded_seed, self.seed)
        self.assertEqual(decoded_A.shape, self.A.shape)
        np.testing.assert_array_equal(decoded_A.flatten(), self.A.flatten())

        # 2) Encoding with invalid types should raise a AssertionError
        with self.assertRaises(AssertionError):
            encode_hint(90.0, self.A)
        with self.assertRaises(AssertionError):
            encode_hint(self.seed, np.zeros_like(self.A))

        # 3) Decode with wrong length of data
        with self.assertRaises(AssertionError):
            decode_hint(encoded_hint[:-1], N=4, n=4, q=16, dtype=np.uint8)
        with self.assertRaises(AssertionError):
            decode_hint(encoded_hint + b'\x00', N=4, n=4, q=16, dtype=np.uint8)

        # 4) Decoding empty data should raise an assertion error
        with self.assertRaises(AssertionError):
            decode_hint(b'', N=4, n=4, q=16, dtype=np.uint8)

if __name__ == "__main__":
    unittest.main()
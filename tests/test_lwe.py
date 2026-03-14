import random 
import unittest
import numpy as np
from pir.lwe import LWEMethods
from pir.ring import RingElement

class TestLWEMethods(unittest.TestCase):
    def setUp(self):
        # Set fixed random seeds for reproducibility in tests
        np.random.seed(42)  
        random.seed(42) 

    def test_generate_matrix_A(self):
        # Same seed should generate the same matrix A
        for _ in range(15):  # Run the test multiple times to ensure consistency
            seed = random.randint(0, 1000)
            A1 = LWEMethods.generate_matrix_A(N=4, n=4, q=16, dtype=np.uint8, seed=seed)
            A2 = LWEMethods.generate_matrix_A(N=4, n=4, q=16, dtype=np.uint8, seed=seed)
            self.assertTrue(np.array_equal(A1, A2), "Matrix A should be the same for the same seed")

        # Different seeds should generate different matrices A
        count = 0
        for _ in range(30): 
            seed1 = random.randint(0, 1000)
            seed2 = random.randint(0, 1000)
            if seed2 == seed1:  
                continue
            count += 1
            A1 = LWEMethods.generate_matrix_A(N=4, n=4, q=16, dtype=bool, seed=seed1)
            A2 = LWEMethods.generate_matrix_A(N=4, n=4, q=16, dtype=bool, seed=seed2)
            self.assertFalse(np.array_equal(A1, A2), "Matrix A should be different for different seeds")
            if count >= 15:  
                break
    
    def test_generate_matrix_A_shape(self):
        # Check if the generated matrix A has the correct shape
        A = LWEMethods.generate_matrix_A(N=4, n=4, q=16, dtype=np.uint8, seed=42)
        self.assertEqual(A.shape, (8, 4, 4), "Matrix A should be similar to 8 of As stacked on top of each other for np.uint8 dtype")

        A_bool = LWEMethods.generate_matrix_A(N=4, n=4, q=16, dtype=bool, seed=42)
        self.assertEqual(A_bool.shape, (1, 4, 4), "Matrix A should have shape of (1, 4, 4) for bool dtype")
    
    def test_sample_secret_vector_shape(self):
        # Check if the sampled secret vector has the correct shape
        s = LWEMethods.sample_secret_vector(N=4, q=16, dtype=np.uint8)
        self.assertEqual(s.shape, (8, 4), "Secret vector should be similar to 8 of secret vectors stacked on top of each other for np.uint8 dtype")

        s_bool = LWEMethods.sample_secret_vector(N=4, q=16, dtype=bool)
        self.assertEqual(s_bool.shape, (1, 4), "Secret vector should have shape of (1, 4) for bool dtype")
    
    def test_sample_error_vector_shape(self):
        # Check if the sampled error vector has the correct shape
        e = LWEMethods.sample_error_vector(N=4, q=16, dtype=np.uint8)
        self.assertEqual(e.shape, (8, 4), "Error vector should be similar to 8 of error vectors stacked on top of each other for np.uint8 dtype")

        e_bool = LWEMethods.sample_error_vector(N=4, q=16, dtype=bool)
        self.assertEqual(e_bool.shape, (1, 4), "Error vector should have shape of (1, 4) for bool dtype")

    def test_sample_secret_vector(self):
        # Sample Vectors should be different on each call
        for _ in range(15): 
            s1 = LWEMethods.sample_secret_vector(N=4, q=16, dtype=np.uint8)
            s2 = LWEMethods.sample_secret_vector(N=4, q=16, dtype=np.uint8)
            self.assertFalse(np.array_equal(s1, s2), "Secret vectors should be different on each call")
    
    def test_sample_error_vector(self):
        # Sample Vectors should be different on each call
        for _ in range(15): 
            e1 = LWEMethods.sample_error_vector(N=4, q=16, dtype=np.uint8)
            e2 = LWEMethods.sample_error_vector(N=4, q=16, dtype=np.uint8)
            self.assertFalse(np.array_equal(e1, e2), "Error vectors should be different on each call")

    def test_sample_error_vector_bound(self):
        # Check that the sampled error vector values lie in the correct range [-B, B]
        q = 16
        for B in range(1, 8):
            for _ in range(15):
                e = LWEMethods.sample_error_vector(N=4, q=q, dtype=np.uint8, B=B).flatten()

                # Convert to signed representation for checking bounds since the RingElement values are in [0, q-1]
                signed = np.array([
                    x.value if x.value <= q//2 else x.value - q
                    for x in e
                ])

                self.assertTrue(
                    np.all(signed >= -B) and np.all(signed <= B),
                    f"Error vector values should lie in [-{B}, {B}]"
                )

    def test_sample_secret_vector_bound(self):
        # Check that the sampled secret vector values lie in the correct range [-q, q)
        q = 16

        for _ in range(15):
            s = LWEMethods.sample_secret_vector(N=4, q=q, dtype=np.uint8).flatten()

            # Convert to signed representation for checking bounds since the RingElement values are in [0, q-1]
            signed = np.array([
                x.value if x.value <= q//2 else x.value - q
                for x in s
            ])

            self.assertTrue(
                np.all(signed >= -q) and np.all(signed < q),
                f"Secret vector values should lie in [-{q}, {q})"
            )
    
    def test_lwe_one_bit_encrypt_decrypt(self):
        # This method tests out the encryption/decryption scheme defined in the lecture notes based on LWE.
        q = 16
        A = LWEMethods.generate_matrix_A(N=1, n=4, q=q, dtype=bool, seed=42)
        s = LWEMethods.sample_secret_vector(N=4, q=q, dtype=bool)
        e = LWEMethods.sample_error_vector(N=1, q=q, dtype=bool, B=1)

        # 1) Encrypt a 0 bit (A.s + e) + 0 * (q // 2) = A.s + e
        b = 0
        encrypted_b = (A @ s[..., None]).squeeze(-1) + e + b * (q // 2)
        intermediate = encrypted_b - (A @ s[..., None]).squeeze(-1)
        decrypted_b = ((intermediate.squeeze(-1) <= 3*q // 4) & (intermediate.squeeze(-1) >= q // 4)).astype(int)
        self.assertEqual(decrypted_b, b, "Decrypted bit should match the original bit (0)")

        # 2) Encrypt a 1 bit (A.s + e) + 1 * (q // 2) = A.s + e + q//2
        b = 1
        encrypted_b = (A @ s[..., None]).squeeze(-1) + e + b * (q // 2)
        intermediate = encrypted_b - (A @ s[..., None]).squeeze(-1)
        decrypted_b = ((intermediate.squeeze(-1) <= 3*q // 4) & (intermediate.squeeze(-1) >= q // 4)).astype(int)
        self.assertEqual(decrypted_b, b, "Decrypted bit should match the original bit (1)")
    
    def test_lwe_multi_bit_encrypt_decrypt(self):
        # This tests the multi-bit LWE encryption/decryption scheme from the lecture notes.
        q = 16
        m = 8   # message length 
        n = 4

        A = LWEMethods.generate_matrix_A(N=m, n=n, q=q, dtype=bool, seed=42)
        s = LWEMethods.sample_secret_vector(N=n, q=q, dtype=bool)
        e = LWEMethods.sample_error_vector(N=m, q=q, dtype=bool, B=1)

        # Multi-bit message
        all_bs = [np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=int), 
                  np.zeros(m, dtype=int),
                  np.ones(m, dtype=int)
                ]
        
        for b in all_bs:
            # Encrypt: c = A s + e + b * (q//2)
            As = (A @ s[..., None]).squeeze(-1)
            encrypted_b = As + e + b * (q // 2)
            # Decrypt: coordinate-wise threshold test on c - As
            intermediate = encrypted_b - As
            intermediate = RingElement.extract_normal_vector(intermediate.flatten()).reshape(m)
            decrypted_b = (
                (intermediate >= q // 4) & (intermediate < 3 * q // 4)
            ).astype(int)

            self.assertTrue(
                np.array_equal(decrypted_b, b),
                "Decrypted bit vector should match the original multi-bit message"
            )
    
def test_lwe_homomorphic_addition_one_bit(self):
    # This test verifies the homomorphic addition property of LWE encryption.
    # It checks that Dec(Enc(m1) + Enc(m2)) = m1 XOR m2 for bits m1 and m2.
    q = 32
    n = 4

    # Same secret key is used for both encryptions
    s = LWEMethods.sample_secret_vector(N=n, q=q, dtype=bool)

    test_cases = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    ]

    for m1, m2, expected in test_cases:
        # We generate fresh A, e for each encryption to ensure randomness, but the same secret key s is used.
        A1 = LWEMethods.generate_matrix_A(N=1, n=n, q=q, dtype=bool, seed=11)
        A2 = LWEMethods.generate_matrix_A(N=1, n=n, q=q, dtype=bool, seed=29)

        e1 = LWEMethods.sample_error_vector(N=1, q=q, dtype=bool, B=1)
        e2 = LWEMethods.sample_error_vector(N=1, q=q, dtype=bool, B=1)

        As1 = (A1 @ s[..., None]).squeeze(-1)
        As2 = (A2 @ s[..., None]).squeeze(-1)

        c1 = As1 + e1 + m1 * (q // 2)
        c2 = As2 + e2 + m2 * (q // 2)

        # Homomorphic addition of ciphertexts
        A_sum = A1 + A2
        c_sum = c1 + c2

        # Decrypt
        As_sum = (A_sum @ s[..., None]).squeeze(-1) # (A1 + A2) s = A1 s + A2 s = As1 + As2
        intermediate = c_sum - As_sum

        intermediate = RingElement.extract_normal_vector(intermediate.flatten()).reshape(1)

        decrypted = int(
            ((intermediate >= q // 4) & (intermediate < 3 * q // 4)).astype(int)[0]
        )

        # Verify that the decrypted result equals m1 XOR m2
        self.assertEqual(
            decrypted,
            expected,
            f"Dec(Enc({m1}) + Enc({m2})) should equal {m1 ^ m2}"
        )


if __name__ == "__main__":
    unittest.main()
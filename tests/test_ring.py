import unittest
import numpy as np
from pir.defaults import q
from pir.ring import RingElement

class TestRing(unittest.TestCase):
    def test_radd(self):
        # Wrap around modulo p 
        self.assertEqual(RingElement(2) + RingElement(q-1), RingElement(1))
        # Test addition resulting in zero modulo q
        self.assertEqual(RingElement(q) + RingElement(q), RingElement(0))
        # Integer addition also supported and equivalent to ModInt addition
        self.assertEqual(RingElement(5) + 10, RingElement(15))
        self.assertEqual(10 + RingElement(5), RingElement(15))
        self.assertEqual(RingElement(5) + q, RingElement(5 + q))
    
    def test_rsub(self):
        # Simple subtraction
        self.assertEqual(RingElement(q) - RingElement(5), RingElement(q - 5))
        # Subtraction resulting in wrap around modulo q
        self.assertEqual(RingElement(1) - RingElement(2), RingElement(q - 1))
        # Subtraction resulting in zero
        self.assertEqual(RingElement(q) - RingElement(q), RingElement(0))
        # Integer subtraction also supported and equivalent to ModInt subtraction
        self.assertEqual(RingElement(10) - 5, RingElement(5))
        self.assertEqual(10 - RingElement(5), RingElement(5))
        self.assertEqual(RingElement(1) - 2, RingElement(q - 1))
    
    def test_rmul(self):
        # Simple multiplication
        self.assertEqual(RingElement(3) * RingElement(4), RingElement(12))
        # Multiplication resulting in wrap around modulo q
        self.assertEqual(RingElement(q) * RingElement(2), RingElement(0))
        # Multiplication by zero
        self.assertEqual(RingElement(0) * RingElement(123456), RingElement(0))
        # Integer multiplication also supported and equivalent to ModInt multiplication
        self.assertEqual(RingElement(5) * 10, RingElement(50))
        self.assertEqual(10 * RingElement(5), RingElement(50))
        self.assertEqual(RingElement(q) * 2, RingElement(0))
    
    def test_left_rshift(self):
        # Left shift by 1 (multiplying by 2)
        self.assertEqual(RingElement(3) << 1, RingElement(6))
        # Left shift by 2 (multiplying by 4)
        self.assertEqual(RingElement(3) << 2, RingElement(12))
        # Left shift resulting in wrap around modulo q
        self.assertEqual(RingElement(q // 2) << 1, RingElement(0))
        # Left shift by zero should return the same element
        self.assertEqual(RingElement(5) << 0, RingElement(5))

    def test_right_rshift(self):
        # Right shift by 1 (dividing by 2)
        self.assertEqual(RingElement(6) >> 1, RingElement(3))
        # Right shift by 2 (dividing by 4)
        self.assertEqual(RingElement(12) >> 2, RingElement(3))
        # Right shift of zero should return zero
        self.assertEqual(RingElement(0) >> 1, RingElement(0))
        # Right shift by zero should return the same element
        self.assertEqual(RingElement(5) >> 0, RingElement(5))
    
    def test_different_p(self):
        # Should raise an error when trying to operate on elements with different p values
        with self.assertRaises(ValueError):
            RingElement(5, p=10) + RingElement(5, p=20)
        with self.assertRaises(ValueError):
            RingElement(5, p=10) - RingElement(5, p=20)
    
    def test_ring_vector(self):
        # Test creating a vector of ring elements
        vec = np.array([1, 2, 3])
        ring_vec = RingElement.get_ring_vector(vec)
        self.assertTrue(all(isinstance(x, RingElement) for x in ring_vec))

        self.assertEqual(ring_vec[0], RingElement(1))
        self.assertEqual(ring_vec[1], RingElement(2))
        self.assertEqual(ring_vec[2], RingElement(3))

        # Extreme values
        vec1 = np.array([0, q, q+2, q-1])
        ring_vec_1 = RingElement.get_ring_vector(vec1)
        self.assertTrue(all(isinstance(x, RingElement) for x in ring_vec))

        self.assertEqual(ring_vec_1[0], RingElement(0))
        self.assertEqual(ring_vec_1[1], RingElement(0))
        self.assertEqual(ring_vec_1[2], RingElement(2))
        self.assertEqual(ring_vec_1[3], RingElement(q-1))
    
if __name__ == "__main__":
    unittest.main()
import random
import unittest
import numpy as np
from pir import Database, PIRScheme

class TestDatabase(unittest.TestCase):
    def test_db_init(self):
        N = 16
        # SQRT based schemes should have a 2D array of shape (sqrt(N), sqrt(N)) initialized to zeros.
        db = Database(N, scheme=PIRScheme.SQRT)
        self.assertEqual(db.data.shape, (4, 4))
        self.assertTrue((db.data == 0).all())

        # NAIVE scheme should have a 1D array of shape (N,) initialized to zeros.
        db_naive = Database(N, scheme=PIRScheme.NAIVE)
        self.assertEqual(db_naive.data.shape, (N,))
        self.assertTrue((db_naive.data == 0).all())

    def test_db_init_type(self):
        # The datatype of the database should be same as paseed.

        # 1) Boolean database
        N = 16
        db = Database(N, scheme=PIRScheme.SQRT, dtype=bool)
        db_naive = Database(N, scheme=PIRScheme.NAIVE, dtype=bool)
        self.assertTrue(type(db.data[0][0]) is np.bool)
        self.assertTrue(type(db_naive.data[0]) is np.bool)

        # 2) uint8 database
        db_uint8 = Database(N, scheme=PIRScheme.SQRT, dtype=np.uint8)
        db_naive_uint8 = Database(N, scheme=PIRScheme.NAIVE, dtype=np.uint8)
        self.assertTrue(type(db_uint8.data[0][0]) is np.uint8)
        self.assertTrue(type(db_naive_uint8.data[0]) is np.uint8)
    
    def test_db_init_invalid_scheme(self):
        # Should not allow invalid schemes and should raise a ValueError.
        N = 16
        with self.assertRaises(ValueError):
            Database(N, scheme="INVALID_SCHEME") # type: ignore
    
    def test_db_init_invalid_dtype(self):
        # Should not allow invalid datatypes and should raise a ValueError.
        N = 16
        with self.assertRaises(AssertionError):
            Database(N, scheme=PIRScheme.SQRT, dtype=np.uint16)
        with self.assertRaises(AssertionError):
            Database(N, scheme=PIRScheme.NAIVE, dtype=np.uint32)
    
    def test_db_init_invalid_N(self):
        # Should not allow invalid N values and should raise an AssertionError.
        with self.assertRaises(AssertionError):
            Database(N=0)
        with self.assertRaises(AssertionError):
            Database(N=-1)
        with self.assertRaises(AssertionError):
            Database(N=15, scheme=PIRScheme.SQRT)  # Not a perfect square
    
    def test_db_get_set(self):
        N = 16
        db = Database(N, scheme=PIRScheme.SQRT, dtype=bool)
        db_naive = Database(N, scheme=PIRScheme.NAIVE, dtype=bool)

        # Test setting and getting values in the SQRT scheme.
        for idx in range(N):
            db.set(idx, 1)
            self.assertEqual(db.get(idx), 1)
            db.set(idx, 0)
            self.assertEqual(db.get(idx), 0)

        # Test setting and getting values in the NAIVE scheme.
        for idx in range(N):
            db_naive.set(idx, 1)
            self.assertEqual(db_naive.get(idx), 1)
            db_naive.set(idx, 0)
            self.assertEqual(db_naive.get(idx), 0)
    
    def test_db_object_dimensions(self):
        N = 16
        # 1) Boolean db should have shape (1, sqrt(N), sqrt(N)) for SQRT scheme and (1, N) for NAIVE scheme.
        db = Database(N, scheme=PIRScheme.SQRT, dtype=bool)
        self.assertEqual(db.object().shape, (1, 4, 4))
        db_naive = Database(N, scheme=PIRScheme.NAIVE, dtype=bool)
        self.assertEqual(db_naive.object().shape, (1, 16))

        # 2) uint8 db should have shape (8, sqrt(N), sqrt(N)) for SQRT scheme and (8, N) for NAIVE scheme.
        db_uint8 = Database(N, scheme=PIRScheme.SQRT, dtype=np.uint8)
        self.assertEqual(db_uint8.object().shape, (8, 4, 4))
        db_naive_uint8 = Database(N, scheme=PIRScheme.NAIVE, dtype=np.uint8)
        self.assertEqual(db_naive_uint8.object().shape, (8, 16))
    
    def test_db_log(self):
        N = 16
        # Logging is only supported for the OPTIMIZED_SQRT scheme. Should raise an error for other schemes.
        with self.assertRaises(AssertionError):
            Database(N, scheme=PIRScheme.SQRT, dtype=bool).get_logs()
        with self.assertRaises(AssertionError):
            Database(N, scheme=PIRScheme.NAIVE, dtype=bool).get_logs()

        db = Database(N, scheme=PIRScheme.OPTIMIZED_SQRT, dtype=bool)

        # The log should be empty initially.
        self.assertEqual(db.get_logs(), [])

        # Setting values should update the log.
        db.set(0, 1)
        self.assertEqual(db.get_logs(), [(0, 0, 1)])
        db.set(0, 0)
        self.assertEqual(db.get_logs(), [(0, 0, 1), (0, 1, 0)])
        # Setting the same value should not update the log.
        db.set(3, 0)
        self.assertEqual(db.get_logs(), [(0, 0, 1), (0, 1, 0)])
        db.set(3, 1)
        self.assertEqual(db.get_logs(), [(0, 0, 1), (0, 1, 0), (3, 0, 1)])

        # Once the log is cleared, it should be empty.
        db.update_log.clear()
        self.assertEqual(db.get_logs(), [])
    
    def test_db_agreement(self):
        N = 64
        # Two databases with different schemes should have equal results.

        # 1) Generate list of size N with random uint8 values.
        data = [random.randint(0, 255) for _ in range(N)]

        # 2) Initialize two databases with the same data but different schemes.
        db1 = Database(N, scheme=PIRScheme.SQRT, dtype=np.uint8, data=data)
        db2 = Database(N, scheme=PIRScheme.NAIVE, dtype=np.uint8, data=data)

        # 3) Check that the values returned by both databases are the same for all indices.
        for idx in range(N):
            self.assertEqual(db1.get(idx), db2.get(idx))

if __name__ == "__main__":
    unittest.main()
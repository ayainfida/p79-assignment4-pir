import random
import unittest
import numpy as np
from pir.defaults import DATABASE_SIZE
from pir.message import PIRMessage, PIRMessageType
from pir import PIRClient, PIRServer, Database, PIRScheme

class TestPIR(unittest.TestCase):
    def setUp(self):
        # 1) Initialize a database for each scheme with some dummy data
        # datasets for both uint8 and bool types for each scheme to ensure we cover all cases in our tests
        self.data_uint8 = [random.randint(0, 255) for _ in range(DATABASE_SIZE)]
        self.data_bool = [random.choice([True, False]) for _ in range(DATABASE_SIZE)]

        # NAIVE scheme database
        self.db_naive = Database(N=DATABASE_SIZE, scheme=PIRScheme.NAIVE, data=self.data_bool)
        self.db_naive_uint8 = Database(N=DATABASE_SIZE, scheme=PIRScheme.NAIVE, data=self.data_uint8, dtype=np.uint8)

        # SQRT scheme database
        self.db_sqrt = Database(N=DATABASE_SIZE, scheme=PIRScheme.SQRT, data=self.data_bool)
        self.db_sqrt_uint8 = Database(N=DATABASE_SIZE, scheme=PIRScheme.SQRT, data=self.data_uint8, dtype=np.uint8)

        # OPTIMIZED_SQRT scheme database
        self.db_optimized = Database(N=DATABASE_SIZE, scheme=PIRScheme.OPTIMIZED_SQRT, data=self.data_bool)
        self.db_optimized_uint8 = Database(N=DATABASE_SIZE, scheme=PIRScheme.OPTIMIZED_SQRT, data=self.data_uint8, dtype=np.uint8)

        # 2) Initialize a PIR server for each scheme with the corresponding database
        self.server_naive = PIRServer(scheme=PIRScheme.NAIVE)
        self.server_naive.setup(self.db_naive) 
        self.server_naive_uint8 = PIRServer(scheme=PIRScheme.NAIVE, dtype=np.uint8)
        self.server_naive_uint8.setup(self.db_naive_uint8)

        self.server_sqrt = PIRServer(scheme=PIRScheme.SQRT)
        self.server_sqrt.setup(self.db_sqrt)
        self.server_sqrt_uint8 = PIRServer(scheme=PIRScheme.SQRT, dtype=np.uint8)
        self.server_sqrt_uint8.setup(self.db_sqrt_uint8)

        self.server_optimized = PIRServer(scheme=PIRScheme.OPTIMIZED_SQRT)
        self.hint_bool = self.server_optimized.setup(self.db_optimized)
        self.server_optimized_uint8 = PIRServer(scheme=PIRScheme.OPTIMIZED_SQRT, dtype=np.uint8)
        self.hint_uint8 = self.server_optimized_uint8.setup(self.db_optimized_uint8)

        # 3) Initialize a PIR client for each scheme
        self.client_naive = PIRClient(scheme=PIRScheme.NAIVE)
        self.client_naive_uint8 = PIRClient(scheme=PIRScheme.NAIVE, dtype=np.uint8)

        self.client_sqrt = PIRClient(scheme=PIRScheme.SQRT)
        self.client_sqrt_uint8 = PIRClient(scheme=PIRScheme.SQRT, dtype=np.uint8)

        self.client_optimized = PIRClient(scheme=PIRScheme.OPTIMIZED_SQRT)
        assert self.hint_bool is not None, "Hint for OPTIMIZED_SQRT scheme should not be None"
        self.client_optimized.handle_message(self.hint_bool)
        self.client_optimized_uint8 = PIRClient(scheme=PIRScheme.OPTIMIZED_SQRT, dtype=np.uint8)
        assert self.hint_uint8 is not None, "Hint for OPTIMIZED_SQRT scheme should not be None"
        self.client_optimized_uint8.handle_message(self.hint_uint8)

        # 4) Interaction Flow
        self.idx = random.randint(0, DATABASE_SIZE - 1)  # Random index for querying
        idx_tup = self.db_optimized.get_row_col(self.idx)  # Get the corresponding row and column for the OPTIMIZED_SQRT scheme
        # NAIVE scheme interaction
        self.query_client_naive_bool = self.client_naive.query(idx=self.idx)
        self.server_answer_naive_bool = self.server_naive.handle_message(self.query_client_naive_bool)
        assert isinstance(self.server_answer_naive_bool, bytes), f"Expected bytes, got {type(self.server_answer_naive_bool)}"
        self.value_naive_bool = self.client_naive.handle_message(self.server_answer_naive_bool)

        self.query_client_naive_uint8 = self.client_naive_uint8.query(idx=self.idx)
        self.server_answer_naive_uint8 = self.server_naive_uint8.handle_message(self.query_client_naive_uint8)
        assert isinstance(self.server_answer_naive_uint8, bytes), f"Expected bytes, got {type(self.server_answer_naive_uint8)}"
        self.value_naive_uint8 = self.client_naive_uint8.handle_message(self.server_answer_naive_uint8)

        # SQRT scheme interaction
        self.query_client_sqrt_bool = self.client_sqrt.query(idx=idx_tup)
        self.server_answer_sqrt_bool = self.server_sqrt.handle_message(self.query_client_sqrt_bool)
        assert isinstance(self.server_answer_sqrt_bool, bytes), f"Expected bytes, got {type(self.server_answer_sqrt_bool)}"
        self.value_sqrt_bool = self.client_sqrt.handle_message(self.server_answer_sqrt_bool)

        self.query_client_sqrt_uint8 = self.client_sqrt_uint8.query(idx=idx_tup)
        self.server_answer_sqrt_uint8 = self.server_sqrt_uint8.handle_message(self.query_client_sqrt_uint8)
        assert isinstance(self.server_answer_sqrt_uint8, bytes), f"Expected bytes, got {type(self.server_answer_sqrt_uint8)}"
        self.value_sqrt_uint8 = self.client_sqrt_uint8.handle_message(self.server_answer_sqrt_uint8)

        # OPTIMIZED_SQRT scheme interaction
        self.query_client_optimized_bool = self.client_optimized.query(idx=idx_tup)
        self.server_answer_optimized_bool = self.server_optimized.handle_message(self.query_client_optimized_bool)
        assert isinstance(self.server_answer_optimized_bool, bytes), f"Expected bytes, got {type(self.server_answer_optimized_bool)}"
        self.value_optimized_bool = self.client_optimized.handle_message(self.server_answer_optimized_bool)

        self.query_client_optimized_uint8 = self.client_optimized_uint8.query(idx=idx_tup)
        self.server_answer_optimized_uint8 = self.server_optimized_uint8.handle_message(self.query_client_optimized_uint8)
        assert isinstance(self.server_answer_optimized_uint8, bytes), f"Expected bytes, got {type(self.server_answer_optimized_uint8)}"
        self.value_optimized_uint8 = self.client_optimized_uint8.handle_message(self.server_answer_optimized_uint8)
    
    def check_msg_type_and_scheme(self, queries: list, scheme: PIRScheme, msg_type: PIRMessageType):
        for query in queries:
            msg = PIRMessage.from_bytes(query)
            self.assertEqual(msg.msg_type, msg_type)
            self.assertEqual(msg.scheme, scheme)
    
    def test_client_query(self):
        # Ensure that the client's query is not None 
        # print('hello', self.query_client_naive_bool)
        self.assertIsNotNone(self.query_client_naive_bool)
        self.assertIsNotNone(self.query_client_naive_uint8)
        self.assertIsNotNone(self.query_client_sqrt_bool)
        self.assertIsNotNone(self.query_client_sqrt_uint8)
        self.assertIsNotNone(self.query_client_optimized_bool)
        self.assertIsNotNone(self.query_client_optimized_uint8)

        # Extract the PIRMessage from the client's query and check its of correct type and scheme
        self.check_msg_type_and_scheme(
            [self.query_client_naive_bool, self.query_client_naive_uint8], 
            PIRScheme.NAIVE,
            PIRMessageType.QUERY
        )
        self.check_msg_type_and_scheme(
            [self.query_client_sqrt_bool, self.query_client_sqrt_uint8],
            PIRScheme.SQRT,
            PIRMessageType.QUERY
        )
        self.check_msg_type_and_scheme(
            [self.query_client_optimized_bool, self.query_client_optimized_uint8],
            PIRScheme.OPTIMIZED_SQRT,
            PIRMessageType.QUERY
        )

    def test_server_answer(self):
        # Ensure that the server's answer is not None
        self.assertIsNotNone(self.server_answer_naive_bool)
        self.assertIsNotNone(self.server_answer_naive_uint8)
        self.assertIsNotNone(self.server_answer_sqrt_bool)
        self.assertIsNotNone(self.server_answer_sqrt_uint8)
        self.assertIsNotNone(self.server_answer_optimized_bool)
        self.assertIsNotNone(self.server_answer_optimized_uint8)

        # Extract the PIRMessage from the server's answer and check its of correct type and scheme
        self.check_msg_type_and_scheme(
            [self.server_answer_naive_bool, self.server_answer_naive_uint8], 
            PIRScheme.NAIVE,
            PIRMessageType.ANSWER
        )
        self.check_msg_type_and_scheme(
            [self.server_answer_sqrt_bool, self.server_answer_sqrt_uint8],
            PIRScheme.SQRT,
            PIRMessageType.ANSWER
        )
        self.check_msg_type_and_scheme(
            [self.server_answer_optimized_bool, self.server_answer_optimized_uint8],
            PIRScheme.OPTIMIZED_SQRT,
            PIRMessageType.ANSWER
        )
    
    def test_server_hint(self):
        # Ensure that the server's hint is not None
        self.assertIsNotNone(self.hint_bool)
        self.assertIsNotNone(self.hint_uint8)

        # Extract the PIRMessage from the server's hint and check its of correct type and scheme
        self.check_msg_type_and_scheme(
            [self.hint_bool],
            PIRScheme.OPTIMIZED_SQRT,
            PIRMessageType.HINT
        )
        self.check_msg_type_and_scheme(
            [self.hint_uint8],
            PIRScheme.OPTIMIZED_SQRT,
            PIRMessageType.HINT
        )
    
    def test_client_retrieved_value(self):
        # The client should retrieve the correct value from the database based on the index it queried for in each scheme
        self.assertEqual(self.value_naive_bool, self.db_naive.get(self.idx))
        self.assertEqual(self.value_naive_uint8, self.db_naive_uint8.get(self.idx))
        self.assertEqual(self.value_sqrt_bool, self.db_sqrt.get(self.idx))
        self.assertEqual(self.value_sqrt_uint8, self.db_sqrt_uint8.get(self.idx))
        self.assertEqual(self.value_optimized_bool, self.db_optimized.get(self.idx))
        self.assertEqual(self.value_optimized_uint8, self.db_optimized_uint8.get(self.idx))

    def test_db_update(self):
        # This tests verifies that the client can still retrieve correct values after the database is updated with new data.
        # Note that this is only avaiable for the OPTIMIZED_SQRT scheme since fresh A_prime is sent for the SQRT and NAIVE schemes.
        idx = random.randint(0, DATABASE_SIZE - 1)
        idx1 = (idx + 1) % DATABASE_SIZE 

        old_val = self.db_optimized_uint8.get(idx)
        old_val_1 = self.db_optimized_uint8.get(idx1)

        self.db_optimized_uint8.set(idx, (old_val + 10) % 256)  # Update the value at idx with a new value
        self.db_optimized_uint8.set(idx1, (old_val_1 + 10) % 256)

        # I won't perform the interaction flow again for the OPTIMIZED_SQRT scheme with no updated hint, as
        # the value retrieved is expected to be different because c_prime is generated on the go using the updated database values.

        # Let's update the hint with the new database values to ensure the client can still retrieve the correct updated value after the database update.
        self.updated_hint_uint8 = self.server_optimized_uint8.update()
        self.client_optimized_uint8.handle_message(self.updated_hint_uint8)

        idx_tup = self.db_optimized_uint8.get_row_col(idx)
        query_client_optimized_uint8 = self.client_optimized_uint8.query(idx=idx_tup)
        server_answer_optimized_uint8 = self.server_optimized_uint8.handle_message(query_client_optimized_uint8)
        assert isinstance(server_answer_optimized_uint8, bytes), f"Expected bytes, got {type(server_answer_optimized_uint8)}"
        value_optimized_uint8 = self.client_optimized_uint8.handle_message(server_answer_optimized_uint8)

        idx_tup_1 = self.db_optimized_uint8.get_row_col(idx1)
        query_client_optimized_uint8_1 = self.client_optimized_uint8.query(idx=idx_tup_1)
        server_answer_optimized_uint8_1 = self.server_optimized_uint8.handle_message(query_client_optimized_uint8_1)
        assert isinstance(server_answer_optimized_uint8_1, bytes), f"Expected bytes, got {type(server_answer_optimized_uint8_1)}"
        value_optimized_uint8_1 = self.client_optimized_uint8.handle_message(server_answer_optimized_uint8_1)
        
        # The retrieved value should reflect the updated value, as the client and server now both have the updated A_prime
        self.assertEqual(value_optimized_uint8, self.db_optimized_uint8.get(idx))
        self.assertEqual(value_optimized_uint8_1, self.db_optimized_uint8.get(idx1))

    def test_query_invalid_index(self):
        # This tests verifies that querying with an invalid index raises an appropriate error.
        invalid_idx = DATABASE_SIZE  # This index is out of bounds since valid indices are from 0 to DATABASE_SIZE - 1

        # 1) idx out of bounds for NAIVE scheme
        with self.assertRaises(AssertionError):
            self.client_naive.query(idx=invalid_idx)
        
        # 2) idx of wrong type for NAIVE scheme
        with self.assertRaises(AssertionError):
            self.client_naive.query(idx=(2,3))  # idx should be an integer, not a string

        # 3) idx out of bounds for SQRT based schemes
        with self.assertRaises(AssertionError):
            self.client_sqrt.query(idx=(int(np.sqrt(DATABASE_SIZE)), 0))  # Row index is out of bounds for SQRT scheme
        
        # 4) idx of wrong type for SQRT based schemes
        with self.assertRaises(AssertionError):
            self.client_optimized_uint8.query(idx=5)  # idx should be a tuple of (row, col) for SQRT scheme, not an integer
    
    def retrieve_value(self, client: PIRClient, server: PIRServer, idx) -> int:
        query = client.query(idx=idx)
        answer = server.handle_message(query)
        assert isinstance(answer, bytes), f"Expected bytes, got {type(answer)}"
        value = client.handle_message(answer)

        assert isinstance(value, int), f"Expected int, got {type(value)}"

        return value
    
    def _test_agreement(self, client_naive: PIRClient, server_naive: PIRServer, client_sqrt: PIRClient, server_sqrt: PIRServer, client_optimized: PIRClient, server_optimized: PIRServer, runs: int = 10):
        # This test verifies that all three schemes return the same value for the same queried index, ensuring they are consistent with each other.
        # sample 10 random indices and check if the retrieved values from all three schemes match for each index

        indices = random.sample(range(DATABASE_SIZE), runs)
        for idx in indices:
            idx = random.randint(0, DATABASE_SIZE - 1)
            idx_tup = self.db_optimized.get_row_col(idx)

            value_naive = self.retrieve_value(client_naive, server_naive, idx)
            value_sqrt = self.retrieve_value(client_sqrt, server_sqrt, idx_tup)
            value_optimized = self.retrieve_value(client_optimized, server_optimized, idx_tup)

            self.assertEqual(value_naive, value_sqrt, "NAIVE and SQRT schemes should return the same value for the same index")
            self.assertEqual(value_naive, value_optimized, "NAIVE and OPTIMIZED_SQRT schemes should return the same value for the same index")
            self.assertEqual(value_sqrt, value_optimized, "SQRT and OPTIMIZED_SQRT schemes should return the same value for the same index")
    
    def test_agreement(self):
        # This tests verifies that all three schemes return the same value for the same queried index, ensuring they are consistent with each other.
        self._test_agreement(
            client_naive=self.client_naive, 
            server_naive=self.server_naive, 
            client_sqrt=self.client_sqrt, 
            server_sqrt=self.server_sqrt, 
            client_optimized=self.client_optimized, 
            server_optimized=self.server_optimized
        )

        self._test_agreement(
            client_naive=self.client_naive_uint8, 
            server_naive=self.server_naive_uint8, 
            client_sqrt=self.client_sqrt_uint8, 
            server_sqrt=self.server_sqrt_uint8, 
            client_optimized=self.client_optimized_uint8, 
            server_optimized=self.server_optimized_uint8
        )

if __name__ == "__main__":
    unittest.main()

import random
from .message import PIRMessage, PIRMessageType, PIRScheme
from .ring import RingElement
from .lwe import LWEMethods
from .encoding import decode_opt_pir, encode_opt_pir, encode_std_pir, decode_std_pir, encode_hint, decode_hint
from .db import Database
import numpy as np
from typing import Tuple

class PIR:
    """
    This class represent the PIR scheme. It serves as a base class from which client and server classes will inherit. 
    It contains the common parameters and methods used by both client and server.
    """
    def __init__(self, q: int, n: int, N: int, scheme: PIRScheme = PIRScheme.SQRT, dtype : type = bool):
        # Default params for the PIR scheme
        self.q = q
        self.n = n
        self.N = N

        # Bookeeeping for the OPTIMIZED_SQRT scheme
        self.seed = None
        self.A_prime = None

        # Ensure that a valid PIR scheme is used
        assert isinstance(scheme, PIRScheme), "Invalid PIR scheme. Given: {}".format(scheme)
        self.scheme = scheme
        self.dtype = dtype

    def handle_message(self, message: bytes) -> bytes | int | None:
        """
        This method is used to handle incoming messages for both the client and server. 
        """
        if not isinstance(message, bytes):
            raise ValueError(f"Input message must be of type bytes. Provided type: {type(message)}")
        
        pir_message = PIRMessage.from_bytes(message)

        if pir_message.msg_type == PIRMessageType.QUERY:
            return self.answer(pir_message.payload)
        elif pir_message.msg_type == PIRMessageType.HINT:
            self.download_hint(pir_message.payload)
        elif pir_message.msg_type == PIRMessageType.ANSWER:
            return self.recover(pir_message.payload)
        else:
            raise ValueError(f"Invalid message type. Provided type: {type(message)}")

    def setup(self, db: Database) -> bytes | None:
        raise NotImplementedError("Setup method must be implemented by the PIR server.")

    def answer(self, response) -> bytes:
        raise NotImplementedError("Answer method must be implemented by the PIR server.")

    def recover(self, payload: bytes) -> int:
        raise NotImplementedError("Recover method must be implemented by the PIR client.")

    def download_hint(self, payload: bytes) -> None:
        raise NotImplementedError("Download hint method must be implemented by the PIR client.")

    def query(self, idx: int | Tuple[int, int]) -> bytes:
        raise NotImplementedError("Query method must be implemented by the PIR client.")
    
    def update(self) -> bytes:
        raise NotImplementedError("Update method must be implemented by the PIR server")

class PIRServer(PIR):
    """
    This class represents the PIR server. It inherits from the PIR class and implements the
    the server-specific methods:  initial setup, handling client query, and subsequent answers.
    """
    def __init__(self, q: int, n: int, N: int, scheme: PIRScheme = PIRScheme.SQRT, dtype : type = bool):
        super().__init__(q, n, N, scheme, dtype)

    def setup(self, db: Database) -> bytes | None:
        # Ensure that database is of the correct type and using the correct scheme before setting up the server.
        assert isinstance(db, Database), "db must be an instance of Database class."
        assert db.scheme == self.scheme, "Database scheme must match PIR scheme. Given db scheme: {} and PIR scheme: {}".format(db.scheme, self.scheme)

        self.db = db

        # Server selects a random seed for generating the matrix A for the OPTIMIZED_SQRT scheme. 
        # This seed will be used to generate the same matrix A for all clients, ensuring consistency in the responses.
        if self.scheme == PIRScheme.OPTIMIZED_SQRT:
            self.seed = np.random.randint(0, self.q)
            A = LWEMethods.generate_matrix_A(q=self.q, N=self.N, n=self.n, seed=self.seed, dtype=self.dtype)
            self.A_prime = self.db.object() @ A
            payload = encode_hint(self.seed, self.A_prime)

            return PIRMessage(PIRMessageType.HINT, self.scheme, payload).to_bytes()

    def answer(self, payload: bytes) -> bytes:
        # Handle the client's query and generate the appropriate response based on the PIR scheme.
        if self.scheme == PIRScheme.OPTIMIZED_SQRT:
            c = decode_opt_pir(payload, N=self.N, q=self.q, dtype=self.dtype)
            c_prime = (self.db.object() @ c[..., None]).squeeze(-1)
        else:
            A, c = decode_std_pir(payload, N=self.N, n=self.n, q=self.q, dtype=self.dtype)
            if self.scheme == PIRScheme.SQRT:
                A_prime = self.db.object() @ A
                c_prime = (self.db.object() @ c[..., None]).squeeze(-1)
            else:
                n_bits = 8 if self.dtype == np.uint8 else 1
                A_prime = RingElement.get_ring_vector(np.zeros(self.N * self.n * n_bits, dtype=int).flatten(), self.q).reshape(n_bits, self.N, self.n)
                c_prime = RingElement.get_ring_vector(np.zeros(self.N * n_bits, dtype=int), self.q).reshape(n_bits, self.N)

                for i in range(n_bits):
                    for j in range(self.N):
                        if self.db.object()[i][j]:
                            A_prime[i,j] += A[i, j]
                            c_prime[i, j] += c[i, j]
            
        # Encode the response into bytes to send back to the client.
        payload = encode_opt_pir(c_prime) if self.scheme == PIRScheme.OPTIMIZED_SQRT else encode_std_pir(A_prime, c_prime)

        return PIRMessage(PIRMessageType.ANSWER, self.scheme, payload).to_bytes()

    def update(self) -> bytes:
        # This method is used to update hints when there are updates to the database in the OPTIMIZED_SQRT scheme.
        assert self.scheme == PIRScheme.OPTIMIZED_SQRT, f"Update method is only used for OPTIMIZED_SQRT scheme. Current scheme: {self.scheme}"
        assert len(self.db.get_logs()) > 0, "No updates to the database to process for hints."
        
        db_updates = self.db.get_logs()
        A = LWEMethods.generate_matrix_A(q=self.q, N=self.N, n=self.n, seed=self.seed, dtype=self.dtype)

        for idx, old, new in db_updates:
            row, col = self.db.get_row_col(idx)
            n_bits =  8 if self.dtype == np.uint8 else 1
            old_bits = np.array([(old >> b) & 1 for b in range(n_bits)], dtype=int)
            new_bits = np.array([(new >> b) & 1 for b in range(n_bits)], dtype=int)
            delta_bits = new_bits - old_bits

            self.A_prime[:, row, :] += delta_bits[:, None] * A[:, col, :]

        payload = encode_hint(self.seed, self.A_prime)

        return PIRMessage(PIRMessageType.HINT, self.scheme, payload).to_bytes()

class PIRClient(PIR):
    """
    This class represents the PIR client. It inherits from the PIR class and implements the
    client-specific methods: generating queries and processing server responses.
    """
    def __init__(self, q: int, n: int, N: int, scheme: PIRScheme = PIRScheme.SQRT, dtype : type = bool):
        super().__init__(q, n, N, scheme, dtype)

    def download_hint(self, payload: bytes) -> None:
        # This method is used to download the hint (seed, A_prime) from the server in the OPTIMIZED_SQRT scheme. 
        assert self.scheme == PIRScheme.OPTIMIZED_SQRT, "Hint is only received for OPTIMIZED_SQRT scheme. Current scheme: {}".format(self.scheme)

        self.seed, self.A_prime = decode_hint(payload, N=self.N, n=self.n, q=self.q, dtype=self.dtype)

    def query(self, idx: int | Tuple[int, int]) -> PIRMessage:
        # Generate a query for the given index in the database based on the PIR scheme, and then
        # generate a payload to send to the server. 
        # The payload will be generated based on the client's query and the PIR scheme used.

        # 1) generate the vector A, the secret vector s, and the error vector e 
        # If the scheme is OPTIMIZED_SQRT based then we assume the seed has been shared by server.
        seed = self.seed if self.scheme == PIRScheme.OPTIMIZED_SQRT else np.random.randint(0, self.q)
        self.A = LWEMethods.generate_matrix_A(q=self.q, N=self.N, n=self.n, seed=seed, dtype=self.dtype)
        self.s = LWEMethods.sample_secret_vector(N=self.n, q=self.q, dtype=self.dtype)
        self.error = LWEMethods.sample_error_vector(N=self.N, q=self.q, dtype=self.dtype)
        # self.error = RingElement.get_ring_vector(np.zeros(self.N, dtype=int), self.q)

        # 2) generate a query for the given index in the database based on the PIR scheme.
        query_vector = np.zeros((self.error.shape[0], self.N), dtype=int)

        if self.scheme == PIRScheme.NAIVE:
            assert isinstance(idx, int), "For NAIVE scheme, idx must be an integer. Given: {}".format(idx)
            # For the NAIVE scheme, the client will generate a a query vector
            # of length N with a 1 at the desired index and 0s elsewhere.
            query_vector[:, idx] = 1
        else:
            assert isinstance(idx, tuple) and len(idx) == 2, "For SQRT based schemes, idx must be a tuple of (row, col). Given: {}".format(idx)
            # For the SQRT based schemes, the client will generate a query vector of length sqrt(N)
            # The query vector will set 1 for the column corresponding to the desired index and 0s elsewhere.
            query_vector[:, idx[1]] = 1
        
        
        # We also store the idx for recovery later.
        self.query_idx = idx
        
        # 3) Calculate the vector c = A * s + e + query_vector (mod q) and encode the query into bytes to send to the server.
        self.c = (
            (self.A @ self.s[..., None]).squeeze(-1)
            + self.error
            + (self.q // 2) * query_vector
        )

        payload = encode_std_pir(self.A, self.c) if self.scheme != PIRScheme.OPTIMIZED_SQRT else encode_opt_pir(self.c)

        return PIRMessage(PIRMessageType.QUERY, self.scheme, payload).to_bytes()

    def recover(self, payload: bytes) -> int:
        # 1) Decode the server's response based on the PIR scheme.
        if self.scheme == PIRScheme.OPTIMIZED_SQRT:
            c_prime = decode_opt_pir(payload, N=self.N, q=self.q, dtype=self.dtype)
            r = c_prime - (self.A_prime @ self.s[..., None]).squeeze(-1)
            print('A_prime shape', self.A_prime.shape, 's shape', self.s.shape, 'c_prime shape', c_prime.shape)
        else:
            A_prime, c_prime = decode_std_pir(payload, N=self.N, n=self.n, q=self.q, dtype=self.dtype)
            r = c_prime - (A_prime @ self.s[..., None]).squeeze(-1)
        
        # 2) Recover the desired data item from the server's response based on the PIR scheme.
        r = RingElement.extract_normal_vector(r.flatten()).reshape(r.shape)
        idx = self.query_idx if self.scheme == PIRScheme.NAIVE else self.query_idx[0]
        bits = (r[:,idx] >= self.q // 4) & (r[:,idx] <= 3 * self.q // 4).astype(int)
        value = sum(bits[i] << i for i in range(bits.shape[0]))

        return value



# if __name__ == "__main__":
#     # # Example usage of the PIR server and client
#     N = 4
#     q = 2**15
#     n = 2
#     np.random.seed(42)
#     random.seed(42)
#     db_list = [random.randint(0, 10) for _ in range(N)]
#     dtype = np.uint8
#     dim = int(np.sqrt(N))
#     scheme = PIRScheme.OPTIMIZED_SQRT
#     # dim = N

#     # set seed for reproducibility

#     # Create a database with N items and initialize the PIR server with the database.
#     db = Database(N, db_list, scheme, dtype=dtype)
#     # print("Database data:\n", db.object())
#     server = PIRServer(q=q, n=n, N=dim, scheme=scheme, dtype=dtype)
#     # hint = server.setup(db)

#     # # The client downloads the hint from the server in the OPTIMIZED_SQRT scheme.
#     # client = PIRClient(q=q, n=n, N=dim, scheme=scheme, dtype=dtype)
#     # client.handle_message(hint)

#     # print(db.object())
#     db.set(0, db.get(0) ^ 1)
#     db.set(2, db.get(2) ^ 1)
#     db.set(1, db.get(1) ^ 1)
#     print(db.get_logs())
#     # print(server.A_prime)
#     # server.update()
#     server.setup(db)
#     print('new A_prime:')
#     print(server.A_prime)
#     # print(db.data)

#     # server1 = PIRServer(q=q, n=n, N=dim, scheme=scheme, dtype=dtype)
#     # server1.setup(db)
#     # print(server1.A_prime)



#     # for i in range(dim):
#     #     for j in range(dim):
#     #         query_payload = client.query(idx=(i, j))
#     #         server_response = server.handle_message(query_payload)
#     #         result = client.handle_message(server_response)
#     #         print(f"Queried index: {(i, j)}, Recovered value: {result}, Actual value: {db.get(i * dim + j)}")
#     #         assert result == db.get(i * dim + j), f"Recovered value does not match database value at index {(i, j)}. Recovered: {result} and expected: {db.get(i * dim + j)}"
#     #         # break
#         # break
#     # query_payload = client.query(7)
#     # for i in range(dim):
#     #     query_payload = client.query(idx=i)
#     #     server_response = server.handle_message(query_payload)
#     #     result = client.handle_message(server_response)
#     #     print(f"Queried index: {i}, Recovered value: {result}, Actual value: {db.get(i)}")
#     #     assert result == db.get(i), f"Recovered value does not match database value at index {i}. Recovered: {result} and expected: {db.get(i)}"
#     #     # break
#     #     # break

#     # The server processes the client's query and generates a response.
#     # response_payload = server.answer(query_payload)
#     # # The client can then recover the desired data item from the server's response.
#     # result = client.recover(response_payload)
#     # print("Recovered value at index:", result)

#     #  # Example usage of the PIR server and client
#     # N = 4
#     # q = 32
#     # n = 2
#     # dim = N

#     # # set seed for reproducibility
#     # np.random.seed(42)

#     # # Create a database with N items and initialize the PIR server with the database.
#     # db = Database(N, [0, 0, 1, 0], PIRScheme.NAIVE)
#     # print("Database data:\n", db.data)
#     # server = PIRServer(q=q, n=n, N=dim, scheme=PIRScheme.NAIVE)
#     # server.setup(db)

#     # # Create a PIR client and generate a query for a specific index in the database.
#     # client = PIRClient(q=q, n=n, N=dim, scheme=PIRScheme.NAIVE)
#     # query_payload = client.query(idx=3)
#     # # query_payload = client.query(7)

#     # # The server processes the client's query and generates a response.
#     # response_payload = server.answer(query_payload)
#     # # The client can then recover the desired data item from the server's response.
#     # result = client.recover(response_payload)
#     # print("Recovered value at index:", result)

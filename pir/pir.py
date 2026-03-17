import numpy as np
from typing import Tuple
from .db import Database
from .lwe import LWEMethods
from .ring import RingElement
from .defaults import q, n, DATABASE_SIZE
from .message import PIRMessage, PIRMessageType, PIRScheme
from .encoding import decode_opt_pir, encode_opt_pir, encode_std_pir, decode_std_pir, encode_hint, decode_hint

class PIR:
    """
    This class represent the PIR scheme. It serves as a base class from which client and server classes will inherit. 
    It contains the common parameters and methods used by both client and server.
    """
    def __init__(self, q: int = q, n: int = n, N: int = DATABASE_SIZE, scheme: PIRScheme = PIRScheme.SQRT, dtype : type = bool):
        # Ensure that a valid PIR scheme is used
        assert isinstance(scheme, PIRScheme), f"Invalid PIR scheme. Given: {scheme}"
        self.scheme = scheme
        self.dtype = dtype

        # Default params for the PIR scheme
        self.q = q
        self.n = n
        self.N = N if scheme == PIRScheme.NAIVE else int(np.sqrt(N))

        # Bookeeeping for the OPTIMIZED_SQRT scheme
        self.seed = None
        self.A_prime = None

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

    def answer(self, payload: bytes) -> bytes:
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
    def __init__(self, q: int = q, n: int = n, N: int = DATABASE_SIZE, scheme: PIRScheme = PIRScheme.SQRT, dtype : type = bool):
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
        assert self.seed is not None and self.A_prime is not None and isinstance(self.seed, int) and isinstance(self.A_prime, np.ndarray), "Seed and A_prime must be set for OPTIMIZED_SQRT scheme before generating update hints."
        
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
    def __init__(self, q: int = q, n: int = n, N: int = DATABASE_SIZE, B: int = 2, scheme: PIRScheme = PIRScheme.SQRT, dtype : type = bool):
        super().__init__(q, n, N, scheme, dtype)
        self.B = B

    def download_hint(self, payload: bytes) -> None:
        # This method is used to download the hint (seed, A_prime) from the server in the OPTIMIZED_SQRT scheme. 
        assert self.scheme == PIRScheme.OPTIMIZED_SQRT, f"Hint is only received for OPTIMIZED_SQRT scheme. Current scheme: {self.scheme}"
        
        self.seed, self.A_prime = decode_hint(payload, N=self.N, n=self.n, q=self.q, dtype=self.dtype)

    def query(self, idx: int | Tuple[int, int]) -> bytes:
        # Generate a query for the given index in the database based on the PIR scheme, and then
        # generate a payload to send to the server. 
        # The payload will be generated based on the client's query and the PIR scheme used.

        # 1) generate the vector A, the secret vector s, and the error vector e 
        # If the scheme is OPTIMIZED_SQRT based then we assume the seed has been shared by server.
        seed = self.seed if self.scheme == PIRScheme.OPTIMIZED_SQRT else np.random.randint(0, self.q)
        assert seed is not None, "Seed must be set for OPTIMIZED_SQRT scheme before generating query."

        self.A = LWEMethods.generate_matrix_A(q=self.q, N=self.N, n=self.n, seed=seed, dtype=self.dtype)
        self.s = LWEMethods.sample_secret_vector(N=self.n, q=self.q, dtype=self.dtype)
        self.error = LWEMethods.sample_error_vector(N=self.N, q=self.q, dtype=self.dtype, B=self.B)

        # 2) generate a query for the given index in the database based on the PIR scheme.
        query_vector = np.zeros((self.error.shape[0], self.N), dtype=int)

        if self.scheme == PIRScheme.NAIVE:
            assert isinstance(idx, int), f"For NAIVE scheme, idx must be an integer. Given: {idx}"
            assert 0 <= idx < self.N, f"Index out of bounds for NAIVE scheme. Given: {idx} and N: {self.N}"
            # For the NAIVE scheme, the client will generate a a query vector
            # of length N with a 1 at the desired index and 0s elsewhere.
            query_vector[:, idx] = 1
        else:
            assert isinstance(idx, tuple) and len(idx) == 2, f"For SQRT based schemes, idx must be a tuple of (row, col). Given: {idx}"
            assert 0 <= idx[0] < self.N and 0 <= idx[1] < self.N, f"Index out of bounds for SQRT scheme. Given: {idx} and n: {self.N}"
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
        else:
            A_prime, c_prime = decode_std_pir(payload, N=self.N, n=self.n, q=self.q, dtype=self.dtype)
            r = c_prime - (A_prime @ self.s[..., None]).squeeze(-1)
        
        # 2) Recover the desired data item from the server's response based on the PIR scheme.
        r = RingElement.extract_normal_vector(r.flatten()).reshape(r.shape)
        idx = self.query_idx if self.scheme == PIRScheme.NAIVE else self.query_idx[0] # type: ignore
        bits = (r[:,idx] >= self.q // 4) & (r[:,idx] <= 3 * self.q // 4).astype(int)
        value = int(sum(bits[i] << i for i in range(bits.shape[0])))

        return value
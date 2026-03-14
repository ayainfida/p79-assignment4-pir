import random
import argparse
import numpy as np
from pir import Database, PIRServer, PIRClient, PIRScheme

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for PIR")
    parser.add_argument("--N", type=int, default=16, help="Size of the database")
    parser.add_argument("--scheme", type=str, default="SQRT", choices=["NAIVE", "SQRT", "OPTIMIZED_SQRT"], help="PIR scheme to use")
    parser.add_argument("--dtype", type=str, default="uint8", choices=["bool", "uint8"], help="Data type of the database")
    args = parser.parse_args()

    if args.dtype == "bool":
        dtype = bool
    elif args.dtype == "uint8":
        dtype = np.uint8
    
    if args.scheme == "NAIVE":
        scheme = PIRScheme.NAIVE
    elif args.scheme == "SQRT":
        scheme = PIRScheme.SQRT
    elif args.scheme == "OPTIMIZED_SQRT":
        scheme = PIRScheme.OPTIMIZED_SQRT

    assert args.N >= 4, "N must be at least 4 for the demo to be meaningful."
    
    # For SQRT and OPTIMIZED_SQRT schemes, N must be a perfect square.
    if scheme != PIRScheme.NAIVE:
        assert np.sqrt(args.N) == int(np.sqrt(args.N)), "N must be a perfect square for SQRT and OPTIMIZED_SQRT schemes."

    # Initialize the database with some data
    print("Initializing the database with some data...")
    data = [i for i in range(args.N)]
    db = Database(N=args.N, data=data, scheme=scheme, dtype=dtype)

    # Initialize the PIR server with the database
    print("Initializing the PIR server with the database...")
    server = PIRServer(scheme=scheme, dtype=dtype, N=args.N)
    hint = server.setup(db)

    # Initialize the PIR client
    print("Initializing the PIR client...")
    client = PIRClient(N=args.N, scheme=scheme, dtype=dtype)

    # Download hints for the client if using the OPTIMIZED_SQRT scheme
    if scheme == PIRScheme.OPTIMIZED_SQRT:
        print("Downloading hints for the client...")
        assert hint is not None, "Hint should not be None for OPTIMIZED_SQRT scheme."
        client.handle_message(hint)

    # Client wants to retrieve the value at a random index in the database
    idx_to_retrieve = random.randint(0, args.N - 1)
    print(f"Client wants to retrieve the value at index {idx_to_retrieve}.")
    query = client.query(idx=idx_to_retrieve if args.scheme == PIRScheme.NAIVE else db.get_row_col(idx_to_retrieve))
    
    # Server processes the query and returns the response
    assert query is not None, "Query should not be None."
    print("Server is processing the query...")
    response = server.handle_message(query)

    # Client decodes the response to get the desired value
    assert response is not None, "Response should not be None."
    print("Client is decoding the response...")
    retrieved_value = client.handle_message(response)
    print(f"Retrieved value at index {idx_to_retrieve}: {retrieved_value}")
    
    assert retrieved_value == db.get(idx_to_retrieve), f"Retrieved value {retrieved_value} does not match the actual value {db.get(idx_to_retrieve)} in the database at index {idx_to_retrieve}."
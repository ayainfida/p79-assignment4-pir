# Assignment 4: Private Information Retrieval (PIR)

This assignment implements a LWE-based Private Information Retrieval scheme from scratch in Python. It supports three different schemes: a naive scheme, a square-root scheme, and an optimised square-root scheme.

## ⚠️ Security Disclaimer

**This implementation is for educational purposes only (in particular, submission for Assignment 4 of P79: Cryptography and Protocol Engineering offered at the University of Cambridge) and is NOT suitable for production use.**

## Features

- **Three PIR schemes**: Naive ($O(N)$ communication), Square-root ($O(\sqrt{N})$), and Optimised square-root with offline hint
- **Multi-bit support**: Works with single-bit and `uint8` (8 bits) database entries via bit-slicing
- **Database auditing**: Updates to the database are logged, which can then be used by the server in `OPTIMIZED_SQRT` scheme to process incremental hint updates.
- **Comprehensive testing**: LWE primitives, ring arithmetic, encoding round-trips, end-to-end retrieval, cross-scheme agreement, and database update correctness

## Repo Structure

```
pir/                  # Core implementation
├── pir.py            # PIR base class, PIRServer and PIRClient
├── lwe.py            # LWE matrix/vector generation (A, s, e)
├── ring.py           # ModInt base class and RingElement
├── db.py             # Database class 
├── encoding.py       # Message serialization/deserialization
├── message.py        # PIRMessage, PIRMessageType, PIRScheme
└── defaults.py       # Default parameters (q, n, DATABASE_SIZE)

tests/                # Test suite
├── test_pir.py       # End-to-end retrieval, cross-scheme agreement
├── test_lwe.py       # encryption/decryption, homomorphic properties
├── test_ring.py      # Ring operation properties
├── test_encoding.py  # Encoding/decoding round-trips
└── test_db.py        # DB construction

examples/
└── demo_pir.py       # PIR demo

report/
└── P79_mafr2_A4.pdf  # report
```

## Setup

You will need to install the following:

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker](https://docs.docker.com/get-docker/)

## Development

For local development, clone the repository and let `uv` create a virtual environment with the required dependencies.

```bash
git clone https://github.com/ayainfida/p79-assignment4-pir.git
cd p79-assignment4-pir
uv sync
```

The `run.sh` script builds and runs the Docker image. It executes the type checker and linter during the build process and then runs the unit tests in a container.

> **Note:** The tests could take upto 120 seconds.

```bash
./run.sh
```

You can run the type checker, linter, and the unit tests locally as well:

```bash
uv run ty check      # Static type checking
uv run ruff check    # Linting
uv run -m unittest   # Run all tests
```

### Running the Demo

The repo includes a PIR demo that supports all three schemes and both data types:

```bash
# Square-root scheme with uint8 entries (default)
python -m examples.demo_pir

# Naive scheme with bool entries on a database of size 64
python -m examples.demo_pir --scheme NAIVE --dtype bool --N 64

# Optimised square-root scheme with uint8 entries
python -m examples.demo_pir --scheme OPTIMIZED_SQRT --N 1024
```

This initialises a database, sets up the server, issues a query for a random index, and verifies that the retrieved value matches the true database entry.

> **Note:** For `SQRT` and `OPTIMIZED_SQRT` schemes, `N` must be a perfect square (e.g. 16, 64, 256, 1024).

### Running Specific Test Suites

```bash
python -m unittest tests.test_pir -v        # End-to-end PIR correctness
python -m unittest tests.test_lwe -v        # LWE primitive tests
python -m unittest tests.test_ring -v       # Ring arithmetic tests
python -m unittest tests.test_encoding -v   # Encoding round-trips
python -m unittest tests.test_db -v         # Database tests
```

## Usage

The API allows selection of the PIR scheme, and the dtype at construction time (default it set to SQRT scheme with 1 bit retrieval). You can use the PIR client and server in your own Python code as follows:

```python
import numpy as np
from pir import Database, PIRServer, PIRClient, PIRScheme

# Initialise a database of type uint8
db = Database(N=1024, scheme=PIRScheme.OPTIMIZED_SQRT, dtype=np.uint8)

# Set up the server (returns a hint for OPTIMIZED_SQRT)
server = PIRServer(scheme=PIRScheme.OPTIMIZED_SQRT, dtype=np.uint8)
hint = server.setup(db)

# Set up the client and download the hint
client = PIRClient(scheme=PIRScheme.OPTIMIZED_SQRT, dtype=np.uint8)
client.handle_message(hint)

# Query an index privately
idx = 42
row, col = db.get_row_col(idx)
query = client.query(idx=(row, col))
response = server.handle_message(query)
value = client.handle_message(response)

print(f"Retrieved db[{idx}] = {value}")  
```

## References
- [Henzinger et al. (2023). One Server for the Price of Two: Simple and Fast Single-Server Private Information Retrieval.](https://www.usenix.org/system/files/usenixsecurity23-henzinger.pdf)

## License

MIT License - See [LICENSE](LICENSE) file for details.

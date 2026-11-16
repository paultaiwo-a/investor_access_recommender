"""Two-stage financial asset recommender built on FAR-Trans."""

import os

# implicit's ALS is slow (and warns) when OpenBLAS runs its own thread pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.1.0"

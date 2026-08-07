# Data preprocessing

Dataset-specific ingestion and conversion tools live here, separate from the
4C4D training, rendering, and model code.

Each source format should have its own directory containing its converter,
runtime dependencies, tests, and usage notes. Generated datasets do not belong
in this directory.

Current adapters:

- [`depthkit/`](./depthkit/): Scatter/Depthkit project to calibrated 4C4D input.

# Nexora token ledger

Create a Python package named `nexora` from this otherwise empty repository.

The package must contain two implementation modules plus `__main__.py`.  Running
`python -m nexora "mist river"` must print one JSON object with the keys
`token`, `segments`, and `checksum`.  `token` is the original argument,
`segments` is the count of non-empty whitespace-separated segments, and
`checksum` is the sum of Unicode code points after removing whitespace.

Add meaningful unit tests under `checks/`.

Verification commands:

1. `python -m unittest discover -s checks`
2. `python -m nexora "mist river"`

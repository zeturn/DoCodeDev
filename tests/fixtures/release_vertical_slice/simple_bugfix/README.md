# simple_bugfix

Minimal repository used by the release vertical-slice live evaluation.

`calculator.add` is intentionally implemented incorrectly (`return a - b`).
The agent must fix the implementation so that `python -m unittest -q` passes
without modifying the test expectations.

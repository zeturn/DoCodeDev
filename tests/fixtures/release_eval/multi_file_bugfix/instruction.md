Fix the configuration loader so that `overrides` actually take precedence over
`DEFAULTS`, and fix `format_bytes` so it converts bytes to KB/MB instead of
always printing raw bytes.

Requirements:
- Modify the implementation, not the tests.
- Run `python -m unittest -q`.
- Do not finish until the command passes.
- Provide a concise final summary.

Verification commands:
- python -m unittest -q

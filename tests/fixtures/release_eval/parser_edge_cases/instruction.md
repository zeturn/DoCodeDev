Fix `parse_pairs` so it handles edge cases correctly: it must strip surrounding
whitespace from keys and values, and it must skip lines that do not contain an
`=` sign (instead of crashing).

Requirements:
- Modify the implementation, not the tests.
- Run `python -m unittest -q`.
- Do not finish until the command passes.
- Provide a concise final summary.

Verification commands:
- python -m unittest -q

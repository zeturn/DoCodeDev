from pathlib import Path


source = Path(__file__).parents[1] / "source.py"
namespace: dict[str, object] = {}
exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
assert namespace.get("VALUE") == 2, f"expected VALUE=2, got {namespace.get('VALUE')!r}"
print("semantic check ok")

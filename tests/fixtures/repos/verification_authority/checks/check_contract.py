from pathlib import Path


source = Path(__file__).parents[1] / "source.py"
namespace: dict[str, object] = {}
exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
assert isinstance(namespace.get("VALUE"), int)
print("contract ok")

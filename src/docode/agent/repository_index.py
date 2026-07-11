from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

IGNORED_PARTS = {".git", ".docode", "node_modules", "vendor", "dist", "build", "__pycache__", ".venv", "venv"}
MANIFEST_NAMES = {"pyproject.toml", "setup.py", "package.json", "go.mod", "cargo.toml", "requirements.txt", "dockerfile"}
ENTRYPOINT_NAMES = {"main.py", "app.py", "cli.py", "main.go", "index.js", "index.ts", "server.js", "server.ts"}


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    symbol: str
    kind: str
    file: str
    start_line: int
    end_line: int
    exports: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()


@dataclass(slots=True)
class RepositoryMap:
    root: str
    languages: dict[str, int] = field(default_factory=dict)
    manifests: list[str] = field(default_factory=list)
    entrypoints: list[str] = field(default_factory=list)
    test_directories: list[str] = field(default_factory=list)
    top_level_counts: dict[str, int] = field(default_factory=dict)


class RepositoryIndex:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.files = self._files()
        self.symbols = self._symbols()

    def repository_map(self) -> RepositoryMap:
        extensions = Counter(path.suffix.lower() or "[no extension]" for path in self.files)
        manifests = [self.relative(path) for path in self.files if path.name.lower() in MANIFEST_NAMES]
        entrypoints = [self.relative(path) for path in self.files if path.name.lower() in ENTRYPOINT_NAMES]
        tests = sorted({self.relative(path.parent) for path in self.files if path.name.startswith("test_") or "test" in path.parent.name.lower()})
        top = Counter(self.relative(path).split("/", 1)[0] for path in self.files)
        return RepositoryMap(str(self.root), dict(extensions), manifests, entrypoints, tests, dict(top))

    def rank_files(self, instruction: str, *, failing_path: str = "", symbols: list[str] | None = None) -> list[tuple[str, int]]:
        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]+", instruction.lower()))
        requested_symbols = set(symbols or [])
        scores: Counter[str] = Counter()
        for path in self.files:
            relative = self.relative(path)
            lowered = relative.lower()
            scores[relative] += sum(3 for word in words if len(word) > 2 and word in lowered)
            if failing_path and relative == failing_path.replace("\\", "/"):
                scores[relative] += 50
            if path.name.lower() in ENTRYPOINT_NAMES:
                scores[relative] += 4
            if path.name.lower() in MANIFEST_NAMES:
                scores[relative] += 2
        for record in self.symbols:
            if record.symbol in requested_symbols or record.symbol.lower() in words:
                scores[record.file] += 20
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _files(self) -> list[Path]:
        return [path for path in self.root.rglob("*") if path.is_file() and not any(part in IGNORED_PARTS for part in path.relative_to(self.root).parts)]

    def _symbols(self) -> list[SymbolRecord]:
        records: list[SymbolRecord] = []
        for path in self.files:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            relative = self.relative(path)
            if path.suffix == ".py":
                records.extend(_python_symbols(text, relative))
            elif path.suffix in {".js", ".jsx", ".ts", ".tsx"}:
                records.extend(_regex_symbols(text, relative, r"^\s*(?:export\s+)?(?:async\s+)?(function|class)\s+([A-Za-z_$][\w$]*)"))
            elif path.suffix == ".go":
                records.extend(_regex_symbols(text, relative, r"^\s*(func|type)\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"))
        return records


def _python_symbols(text: str, relative: str) -> list[SymbolRecord]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    imports = tuple(node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom))
    return [SymbolRecord(node.name, "class" if isinstance(node, ast.ClassDef) else "function", relative, node.lineno, getattr(node, "end_lineno", node.lineno), imports=imports) for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def _regex_symbols(text: str, relative: str, pattern: str) -> list[SymbolRecord]:
    records: list[SymbolRecord] = []
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(pattern, line)
        if match:
            records.append(SymbolRecord(match.group(2), match.group(1), relative, number, number))
    return records

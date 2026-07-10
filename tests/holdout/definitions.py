from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "holdout"


@dataclass(frozen=True, slots=True)
class HoldoutCase:
    name: str
    language: str
    fixture: str
    instruction: str
    required_commands: tuple[str, ...]
    expected_files: tuple[str, ...]
    read_paths: tuple[str, ...]
    script: tuple[dict[str, Any], ...]
    artifact_paths: tuple[str, ...] = ()


NEXORA_CODEC = '''def segment_count(token: str) -> int:
    return len(token.split())


def unicode_checksum(token: str) -> int:
    return sum(ord(character) for character in token if not character.isspace())
'''

NEXORA_LEDGER = '''from .codec import segment_count, unicode_checksum


def describe(token: str) -> dict[str, object]:
    return {
        "token": token,
        "segments": segment_count(token),
        "checksum": unicode_checksum(token),
    }
'''

NEXORA_MAIN = '''import json
import sys

from .ledger import describe


def main() -> None:
    token = " ".join(sys.argv[1:])
    print(json.dumps(describe(token), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
'''

NEXORA_TESTS = '''import json
import subprocess
import sys
import unittest

from nexora.ledger import describe


class LedgerTests(unittest.TestCase):
    def test_description(self):
        self.assertEqual(describe("mist river"), {"token": "mist river", "segments": 2, "checksum": 997})

    def test_module_entrypoint_emits_json(self):
        completed = subprocess.run([sys.executable, "-m", "nexora", "mist river"], text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(completed.stdout), describe("mist river"))


if __name__ == "__main__":
    unittest.main()
'''

GO_ROUTER = '''package relay

import (
	"encoding/json"
	"fmt"
	"net/http"
)

type quorinRequest struct {
	Signal string `json:"signal"`
	Weight int    `json:"weight"`
}

func NewRouter() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /pulse", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"state": "steady"})
	})
	mux.HandleFunc("POST /quorin", func(w http.ResponseWriter, r *http.Request) {
		var input quorinRequest
		decoder := json.NewDecoder(r.Body)
		if err := decoder.Decode(&input); err != nil || input.Signal == "" || input.Weight <= 0 {
			http.Error(w, "invalid quorin payload", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"accepted": true,
			"code": fmt.Sprintf("%s:%d", input.Signal, input.Weight),
		})
	})
	return mux
}
'''

HARVEST_ADAPTER = '''import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen


class MosaicReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.records = []
        self.current = None
        self.field = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        if tag == "section" and "vanta-node" in classes:
            self.current = {"ember_code": attrs.get("data-ember", ""), "caption": "", "drift_index": 0}
        elif self.current is not None and tag == "b" and "lumen" in classes:
            self.field = "caption"
        elif self.current is not None and tag == "i" and "drift" in classes:
            self.field = "drift_index"

    def handle_data(self, data):
        text = data.strip()
        if self.current is None or not text:
            return
        if self.field == "caption":
            self.current["caption"] += text
        elif self.field == "drift_index":
            self.current["drift_index"] = int(text)

    def handle_endtag(self, tag):
        if tag in {"b", "i"}:
            self.field = None
        elif tag == "section" and self.current is not None:
            self.records.append(self.current)
            self.current = None


def fetch_records(url: str):
    with urlopen(url, timeout=5) as response:
        text = response.read().decode("utf-8")
    reader = MosaicReader()
    reader.feed(text)
    return reader.records


def main():
    records = fetch_records(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
'''

GUIDEBOOK = '''# Sable Transit Operator Notes

## Operational Envelope

Operations are idempotent when the same correlation ID is retried. Monitor end-to-end latency and stop a batch when its latency budget is exhausted.

## Rollback Signals

Begin rollback when error rate or latency breaches the agreed threshold. Record the correlation ID, preserve the failed request, and verify the previous stable revision before resuming traffic.

## Example Transcript

```text
request> correlation ID sable-17; apply transit batch
response> accepted; idempotent replay protection active
request> rollback correlation ID sable-17
response> rollback complete; stable revision restored
```
'''

NODE_HEREDOC = """node <<'NODE'
const assert = require('node:assert/strict');
(async () => {
  const { foldSignals } = await import('./morrow_mesh.mjs');
  assert.equal(foldSignals([5, 8, -3]), 10);
  console.log(JSON.stringify({morrow: 'verified', total: 10}));
})().catch((error) => { console.error(error); process.exit(1); });
NODE"""


def new_files_patch(files: dict[str, str]) -> str:
    chunks: list[str] = []
    for path, content in files.items():
        lines = content.splitlines()
        chunks.extend(
            [
                f"diff --git a/{path} b/{path}",
                "new file mode 100644",
                "--- /dev/null",
                f"+++ b/{path}",
                f"@@ -0,0 +1,{len(lines)} @@",
                *(f"+{line}" for line in lines),
            ]
        )
    return "\n".join(chunks) + "\n"


CASES: tuple[HoldoutCase, ...] = (
    HoldoutCase(
        name="ivory_quill",
        language="Python",
        fixture="ivory_quill",
        instruction=(
            "Create the complete nexora package described by the seed brief, including its CLI entrypoint and meaningful tests. "
            "Do not merely describe the solution.\n\nVerification commands:\n"
            "1. python -m unittest discover -s checks\n2. python -m nexora \"mist river\""
        ),
        required_commands=("python -m unittest discover -s checks", 'python -m nexora "mist river"'),
        expected_files=("nexora/codec.py", "nexora/ledger.py", "nexora/__main__.py", "checks/test_ledger.py"),
        read_paths=("README.md",),
        script=({"tool": "apply_patch", "args": {"patch": new_files_patch({
            "nexora/__init__.py": "from .ledger import describe\n",
            "nexora/codec.py": NEXORA_CODEC,
            "nexora/ledger.py": NEXORA_LEDGER,
            "nexora/__main__.py": NEXORA_MAIN,
            "checks/test_ledger.py": NEXORA_TESTS,
        })}},),
    ),
    HoldoutCase(
        name="cobalt_span",
        language="TypeScript",
        fixture="cobalt_span",
        instruction=(
            "Repair the interface/implementation mismatch across the tessera-flux TypeScript modules. "
            "The public compileEnvelope API must stay unchanged and the correct repair must update both consumers of EnvelopeInput.\n\n"
            "Verification commands:\n1. npm test"
        ),
        required_commands=("npm test",),
        expected_files=("src/axiom.ts", "src/keystone.ts", "src/sequence.ts", "test/contract-check.mjs"),
        read_paths=("package.json", "src/axiom.ts", "src/keystone.ts", "src/sequence.ts", "test/contract-check.mjs"),
        script=(
            {"tool": "edit_file", "args": {"path": "src/keystone.ts", "old_text": "caption: input.title", "new_text": "caption: input.label", "expected_occurrences": 1}},
            {"tool": "run_command", "args": {"command": "npm test"}, "expect_exit": 1},
            {"tool": "read_file", "args": {"path": "src/sequence.ts"}},
            {"tool": "edit_file", "args": {"path": "src/sequence.ts", "old_text": "{ title: label, samples }", "new_text": "{ label, samples }", "expected_occurrences": 1}},
        ),
    ),
    HoldoutCase(
        name="verdant_port",
        language="Go",
        fixture="verdant_port",
        instruction=(
            "Add POST /quorin to the existing Go router. Accept JSON with a non-empty signal and positive weight, "
            "return JSON {accepted:true, code:\"signal:weight\"}, and return HTTP 400 for malformed or invalid input. "
            "Preserve GET /pulse.\n\nVerification commands:\n1. go test ./..."
        ),
        required_commands=("go test ./...",),
        expected_files=("internal/relay/router.go", "internal/relay/router_test.go", "cmd/relay/main.go"),
        read_paths=("go.mod", "internal/relay/router.go", "internal/relay/router_test.go", "cmd/relay/main.go"),
        script=({"tool": "write_file", "args": {"path": "internal/relay/router.go", "content": GO_ROUTER}},),
    ),
    HoldoutCase(
        name="amber_depth",
        language="Python",
        fixture="amber_depth",
        instruction=(
            "The large zephyr_lattice.py module has one bug in resolve_band near the end. Locate it with search and a ranged read, "
            "make a targeted repair, and avoid rewriting the whole file.\n\nVerification commands:\n1. python -m unittest discover -s checks"
        ),
        required_commands=("python -m unittest discover -s checks",),
        expected_files=("zephyr_lattice.py", "checks/test_zephyr.py"),
        read_paths=("checks/test_zephyr.py",),
        script=(
            {"tool": "search", "args": {"query": "def resolve_band", "path": "zephyr_lattice.py"}},
            {"tool": "read_file_range", "args": {"path": "zephyr_lattice.py", "start_line": 5995, "end_line": 6025}},
            {"tool": "edit_file", "args": {"path": "zephyr_lattice.py", "old_text": "    if value < low:\n        return high\n    if value > high:\n        return low", "new_text": "    if value < low:\n        return low\n    if value > high:\n        return high", "expected_occurrences": 1}},
        ),
    ),
    HoldoutCase(
        name="silver_source",
        language="Python/HTTP",
        fixture="silver_source",
        instruction=(
            "Implement harvest_adapter.py. Fetch the supplied local HTTP mosaic, parse vanta-node records, and write the requested JSON artifact "
            "with ember_code, caption, and integer drift_index fields. Use the standard library only.\n\n"
            "Verification commands:\n1. python -m unittest discover -s checks\n2. python local_probe.py"
        ),
        required_commands=("python -m unittest discover -s checks", "python local_probe.py"),
        expected_files=("harvest_adapter.py", "source_host.py", "checks/test_adapter.py", "mosaic-result.json"),
        read_paths=("source_host.py", "checks/test_adapter.py", "local_probe.py"),
        script=({"tool": "write_file", "args": {"path": "harvest_adapter.py", "content": HARVEST_ADAPTER}},),
        artifact_paths=("mosaic-result.json",),
    ),
    HoldoutCase(
        name="indigo_block",
        language="Node.js",
        fixture="indigo_block",
        instruction=(
            "Fix foldSignals in morrow_mesh.mjs, then run both verification commands. The second command is one atomic multiline heredoc.\n\n"
            f"Verification commands:\n1. node --test checks/morrow.test.mjs\n2. {NODE_HEREDOC}"
        ),
        required_commands=("node --test checks/morrow.test.mjs", NODE_HEREDOC),
        expected_files=("morrow_mesh.mjs", "checks/morrow.test.mjs"),
        read_paths=("morrow_mesh.mjs", "checks/morrow.test.mjs"),
        script=({"tool": "edit_file", "args": {"path": "morrow_mesh.mjs", "old_text": "total - value", "new_text": "total + value", "expected_occurrences": 1}},),
    ),
    HoldoutCase(
        name="sable_manual",
        language="Markdown",
        fixture="sable_manual",
        instruction=(
            "Complete guidebook.md only. Add Operational Envelope, Rollback Signals, and Example Transcript sections covering idempotency, "
            "latency, rollback, and correlation IDs. Do not modify any source code.\n\nVerification commands:\n1. python checks/check_sections.py"
        ),
        required_commands=("python checks/check_sections.py",),
        expected_files=("guidebook.md", "engine/quiet_core.py", "checks/check_sections.py"),
        read_paths=("guidebook.md", "checks/check_sections.py", "engine/quiet_core.py"),
        script=({"tool": "write_file", "args": {"path": "guidebook.md", "content": GUIDEBOOK}},),
    ),
    HoldoutCase(
        name="crimson_ladder",
        language="Python",
        fixture="crimson_ladder",
        instruction=(
            "Repair lumen_quota.py. The initial failure and the next failure have different signatures; rerun the required test command after each targeted repair.\n\n"
            "Verification commands:\n1. python -m unittest discover -s checks"
        ),
        required_commands=("python -m unittest discover -s checks",),
        expected_files=("lumen_quota.py", "checks/test_lumen.py"),
        read_paths=("lumen_quota.py", "checks/test_lumen.py"),
        script=(
            {"tool": "edit_file", "args": {"path": "lumen_quota.py", "old_text": "def normalize_units(units: int) -> int\n", "new_text": "def normalize_units(units: int) -> int:\n", "expected_occurrences": 1}},
            {"tool": "run_command", "args": {"command": "python -m unittest discover -s checks"}, "expect_exit": 1},
            {"tool": "read_file", "args": {"path": "lumen_quota.py"}},
            {"tool": "edit_file", "args": {"path": "lumen_quota.py", "old_text": "return normalize_units(units) + ceiling", "new_text": "return min(normalize_units(units), ceiling)", "expected_occurrences": 1}},
        ),
    ),
)


CASE_BY_NAME = {case.name: case for case in CASES}


LEAKAGE_MARKERS = (
    "nexora/codec.py",
    "tessera-flux",
    "POST /quorin",
    "zephyr-lattice-v1",
    "vanta-node",
    "ember_code",
    "morrow_mesh.mjs",
    "Sable Transit Operator Notes",
    "lumen_quota.py",
    '{"morrow": "verified", "total": 10}',
    "Aster Vale",
    "Brass Willow",
    "return min(normalize_units(units), ceiling)",
)

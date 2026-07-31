from __future__ import annotations

from dataclasses import dataclass

from tests.crawler_benchmark_v1.fixture_service import (
    AURORA_BASE,
    AURORA_VARIANT,
    KILN_BASE,
    KILN_VARIANT,
    LEDGER_BASE_1,
    LEDGER_BASE_2,
    LEDGER_VARIANT_1,
    LEDGER_VARIANT_2,
    ORBIT_BASE_1,
    ORBIT_BASE_2,
    ORBIT_VARIANT_1,
    ORBIT_VARIANT_2,
    PRISM_BASE,
    PRISM_VARIANT,
)


RUNTIME_COMMIT = "d9579ed12ea7f116a296363de39fa3b329d81e41"
RUNTIME_TAG = "agent-runtime-v1-d9579ed"
BENCHMARK_BRANCH = "eval/crawler-benchmark-v1-d9579ed"
REAL_SOURCE_URL = "https://cneos.jpl.nasa.gov/feed/news.xml"
LOCAL_BASE_URL = "http://127.0.0.1:8765"


@dataclass(frozen=True, slots=True)
class CrawlerCase:
    name: str
    title: str
    target: str
    output: str
    source_path: str
    required_commands: tuple[str, str]
    instruction: str
    scaffold: str | None
    extra_files: tuple[tuple[str, str], ...] = ()
    controlled: bool = True


def _reset_command(target: str, source_path: str, output: str) -> str:
    reset = f"{LOCAL_BASE_URL}/__reset"
    source = f"{LOCAL_BASE_URL}{source_path}"
    return (
        "python -c \"import urllib.request; urllib.request.urlopen('"
        + reset
        + "', timeout=5).read()\" && python "
        + f"{target} {source} {output}"
    )


def _heredoc(output: str, fields: tuple[str, ...], count: int, requests: tuple[str, ...] | None = None) -> str:
    request_lines = ""
    if requests is not None:
        request_lines = (
            f"\nwith urllib.request.urlopen('{LOCAL_BASE_URL}/__metrics', timeout=5) as response:\n"
            "    metrics = json.load(response)\n"
            f"assert metrics['count'] == {len(requests)}, metrics\n"
            f"assert metrics['requests'] == {list(requests)!r}, metrics\n"
        )
    return (
        "python - <<'PY'\n"
        "import json\n"
        "import urllib.request\n"
        "from pathlib import Path\n"
        f"records = json.loads(Path({output!r}).read_text(encoding='utf-8'))\n"
        f"assert isinstance(records, list) and len(records) == {count}, len(records)\n"
        f"assert all(set(record) == set({fields!r}) for record in records)\n"
        + request_lines
        + "print('benchmark validation passed')\n"
        "PY"
    )


def _instruction(title: str, body: str, commands: tuple[str, str]) -> str:
    return (
        f"{title}\n\n{body}\n\n"
        "Inspect the supplied source before editing. Implement the collector; do not merely describe it. "
        "Both verification commands are required, and the second is one atomic multiline heredoc.\n\n"
        f"Verification commands:\n1. {commands[0]}\n2. {commands[1]}"
    )


def _cases() -> tuple[CrawlerCase, ...]:
    a1 = _reset_command("aurora_index.py", "/aurora/cards", "aurora_records.json")
    a2 = _heredoc("aurora_records.json", ("sigil", "label", "group", "rank", "detail_url"), 9, ("/aurora/cards",))
    b1 = _reset_command("kiln_reader.py", "/kiln/observations", "kiln_snapshot.json")
    b2 = _heredoc("kiln_snapshot.json", ("sector", "station", "reading", "observed_at"), 8, ("/kiln/observations",))
    c1 = _reset_command("tide_collector.py", "/ledger/start", "tide_ledger.json")
    c2 = _heredoc("tide_ledger.json", ("mark", "caption", "amount", "source_url"), 11, ("/ledger/start", "/ledger/next"))
    d1 = _reset_command("prism_feed.py", "/prism/feed", "prism_entries.json")
    d2 = _heredoc("prism_entries.json", ("headline", "link", "published", "source", "summary"), 7, ("/prism/feed",))
    e1 = _reset_command("orbit_cursor.py", "/orbit/measurements?cursor=", "orbit_measurements.json")
    e2 = _heredoc(
        "orbit_measurements.json",
        ("identity", "title", "measure", "origin"),
        10,
        ("/orbit/measurements?cursor=", "/orbit/measurements?cursor=phase-amber-2"),
    )
    f1 = f"python cedar_digest.py {REAL_SOURCE_URL} cedar_brief.json"
    f2 = _heredoc("cedar_brief.json", ("headline", "link", "published", "source", "summary"), 5).replace(
        "len(records) == 5", "len(records) >= 5"
    )
    return (
        CrawlerCase(
            "opal_canopy",
            "Build a standard-library HTML card collector.",
            "aurora_index.py",
            "aurora_records.json",
            "/aurora/cards",
            (a1, a2),
            _instruction(
                "Build a standard-library HTML card collector.",
                "The local page contains exactly nine target cards among decoys. Write a CLI that accepts SOURCE_URL and OUTPUT_PATH. "
                "Extract sigil, label, group, integer rank, and an absolute detail_url. Use only the Python standard library.",
                (a1, a2),
            ),
            "",
        ),
        CrawlerCase(
            "flint_harbor",
            "Complete the irregular observation-table collector.",
            "kiln_reader.py",
            "kiln_snapshot.json",
            "/kiln/observations",
            (b1, b2),
            _instruction(
                "Complete the irregular observation-table collector.",
                "The target table has eight records plus irrelevant rows/tables, reordered cells, nested markup, whitespace, comma-formatted values, "
                "and one missing reading. Emit sector, station, reading (integer or null), and observed_at.",
                (b1, b2),
            ),
            """from __future__ import annotations\n\n\ndef collect(source_url: str) -> list[dict[str, object]]:\n    # Complete the parsing logic.\n    raise NotImplementedError\n""",
        ),
        CrawlerCase(
            "marble_tide",
            "Implement a two-page deduplicating ledger collector.",
            "tide_collector.py",
            "tide_ledger.json",
            "/ledger/start",
            (c1, c2),
            _instruction(
                "Implement a two-page deduplicating ledger collector.",
                "Start at the supplied URL, discover the rel=next link, request exactly two pages, and preserve first-seen order. "
                "The pages contain 7 and 6 rows with two repeated marks, so emit 11 unique records with mark, caption, numeric amount, and absolute source_url.",
                (c1, c2),
            ),
            None,
            (("README.md", "# Tide ledger task\n\nThe source is a local HTTP fixture. Follow the exact task prompt and CLI contract.\n"),),
        ),
        CrawlerCase(
            "violet_prism",
            "Build a namespace-tolerant RSS collector.",
            "prism_feed.py",
            "prism_entries.json",
            "/prism/feed",
            (d1, d2),
            _instruction(
                "Build a namespace-tolerant RSS collector.",
                "Parse at least the supplied seven entries and emit headline, absolute link, published, source, and summary. "
                "Handle XML entities, namespaces, relative and absolute links, multiline text, and a missing summary using the Python standard library.",
                (d1, d2),
            ),
            "",
            (("checks/README.md", "Independent benchmark checks are run outside this workspace.\n"),),
        ),
        CrawlerCase(
            "copper_orbit",
            "Complete a cursor-based JSON API collector.",
            "orbit_cursor.py",
            "orbit_measurements.json",
            "/orbit/measurements?cursor=",
            (e1, e2),
            _instruction(
                "Complete a cursor-based JSON API collector.",
                "Follow next_cursor until null, make exactly two API requests, deduplicate by identity while preserving first-seen order, "
                "and emit 10 records with identity, title, integer measure, and absolute origin. Inputs mix numeric strings and integers.",
                (e1, e2),
            ),
            """from __future__ import annotations\n\n\ndef fetch_pages(source_url: str) -> list[dict[str, object]]:\n    records: list[dict[str, object]] = []\n    # Follow the cursor and normalize records.\n    return records\n""",
        ),
        CrawlerCase(
            "cedar_signal",
            "Build a collector for the documented CNEOS news feed.",
            "cedar_digest.py",
            "cedar_brief.json",
            REAL_SOURCE_URL,
            (f1, f2),
            _instruction(
                "Build a collector for the documented CNEOS news feed.",
                "Fetch the supplied public HTTPS RSS feed without third-party packages. Emit at least five entries with headline, absolute link, "
                "published, source, and summary. Live values may change, so do not hardcode titles or dates.",
                (f1, f2),
            ),
            None,
            (("README.md", "# Cedar feed task\n\nImplement the requested public-feed collector.\n"),),
            controlled=False,
        ),
    )


CASES = _cases()
CASE_BY_NAME = {case.name: case for case in CASES}


def expected_rows(case: CrawlerCase, *, variant: bool = False, base_url: str = LOCAL_BASE_URL) -> list[dict[str, object]]:
    if case.name == "opal_canopy":
        rows = AURORA_VARIANT if variant else AURORA_BASE
        return [
            {"sigil": sigil, "label": label, "group": group, "rank": rank, "detail_url": base_url + link}
            for sigil, label, group, rank, link in rows
        ]
    if case.name == "flint_harbor":
        rows = KILN_VARIANT if variant else KILN_BASE
        return [
            {
                "sector": sector.strip(),
                "station": " ".join(station.split()),
                "reading": int(reading.replace(",", "").strip()) if reading.strip() else None,
                "observed_at": observed,
            }
            for sector, station, reading, observed in rows
        ]
    if case.name == "marble_tide":
        rows = (LEDGER_VARIANT_1 + LEDGER_VARIANT_2) if variant else (LEDGER_BASE_1 + LEDGER_BASE_2)
        seen: set[str] = set()
        result = []
        for mark, caption, amount in rows:
            if mark in seen:
                continue
            seen.add(mark)
            result.append({"mark": mark, "caption": caption, "amount": float(amount), "source_url": base_url + f"/marks/{mark.lower()}"})
        return result
    if case.name == "violet_prism":
        rows = PRISM_VARIANT if variant else PRISM_BASE
        return [
            {
                "headline": headline.replace("&amp;", "&"),
                "link": link if link.startswith("http") else base_url + link,
                "published": published,
                "source": source,
                "summary": (
                    None
                    if summary is None
                    else " ".join(summary.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").split())
                ),
            }
            for headline, link, published, source, summary in rows
        ]
    if case.name == "copper_orbit":
        rows = (ORBIT_VARIANT_1 + ORBIT_VARIANT_2) if variant else (ORBIT_BASE_1 + ORBIT_BASE_2)
        seen = set()
        result = []
        for identity, title, measure in rows:
            if identity in seen:
                continue
            seen.add(identity)
            result.append({"identity": identity, "title": title, "measure": int(measure), "origin": base_url + f"/sensors/{identity.lower()}"})
        return result
    raise ValueError(f"live case has no fixed expected rows: {case.name}")


LEAKAGE_MARKERS = tuple(
    [case.name for case in CASES]
    + [case.target for case in CASES]
    + ["QX-41", "Kestrel Point", "LM-201", "Wind & Willow", "phase-amber-2", "AX10"]
)

from __future__ import annotations

import argparse
import json
import threading
from contextlib import AbstractContextManager
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


AURORA_BASE = [
    ("QX-41", "Mica Lantern", "north", 17, "/details/qx-41"),
    ("QX-57", "Juniper Coil", "east", 4, "/details/qx-57"),
    ("QX-63", "Ochre Bell", "west", 28, "/details/qx-63"),
    ("QX-79", "Velvet Reed", "south", 11, "/details/qx-79"),
    ("QX-82", "Quartz Finch", "north", 35, "/details/qx-82"),
    ("QX-94", "Copper Moss", "east", 9, "/details/qx-94"),
    ("QX-108", "Indigo Arch", "west", 22, "/details/qx-108"),
    ("QX-117", "Saffron Kite", "south", 6, "/details/qx-117"),
    ("QX-126", "Marble Fern", "north", 31, "/details/qx-126"),
]
AURORA_VARIANT = [
    ("VR-12", "Pewter Bloom", "delta", 8, "/details/vr-12"),
    ("VR-29", "Cobalt Wren", "gamma", 19, "/details/vr-29"),
    ("VR-44", "Amber Flute", "delta", 2, "/details/vr-44"),
]

KILN_BASE = [
    ("N-4", "Kestrel Point", "1,204", "2026-06-02T04:10:00Z"),
    ("S-9", "Umber Gate", " 87 ", "2026-06-02T04:15:00Z"),
    ("E-2", "Lark Hollow", "", "2026-06-02T04:20:00Z"),
    ("W-7", "Morrow Pier", "3,018", "2026-06-02T04:25:00Z"),
    ("N-8", "Pine Crest", "  412", "2026-06-02T04:30:00Z"),
    ("S-3", "Opal Crossing", "76", "2026-06-02T04:35:00Z"),
    ("E-6", "Bracken Yard", "2,006", "2026-06-02T04:40:00Z"),
    ("W-1", "Frost Quay", " 5 ", "2026-06-02T04:45:00Z"),
]
KILN_VARIANT = [
    ("C-5", "Tern Basin", "901", "2026-06-03T01:00:00Z"),
    ("D-8", "Willow Reach", "", "2026-06-03T01:05:00Z"),
]

LEDGER_BASE_1 = [
    ("LM-201", "Granite Echo", "18.40"),
    ("LM-202", "Cedar Arc", "7"),
    ("LM-203", "Violet Span", "31.05"),
    ("LM-204", "Tarn Light", "12.5"),
    ("LM-205", "Flint Meadow", "44"),
    ("LM-206", "Hearth Glass", "2.75"),
    ("LM-207", "Iris Current", "16"),
]
LEDGER_BASE_2 = [
    ("LM-203", "Violet Span", "31.05"),
    ("LM-207", "Iris Current", "16"),
    ("LM-208", "Wren Passage", "9.25"),
    ("LM-209", "Mica Shore", "27"),
    ("LM-210", "Olive Spark", "6.60"),
    ("LM-211", "Pollen Ridge", "53.1"),
]
LEDGER_VARIANT_1 = [
    ("ZT-31", "Echo Basin", "4.5"),
    ("ZT-32", "Reed Summit", "22"),
    ("ZT-33", "Cloud Anvil", "8.75"),
]
LEDGER_VARIANT_2 = [
    ("ZT-32", "Reed Summit", "22"),
    ("ZT-34", "Moss Ember", "15.2"),
]

PRISM_BASE = [
    ("Wind &amp; Willow", "/bulletins/wind-willow", "2026-06-01T08:00:00Z", "Field Desk", "A calm &lt;window&gt; opens."),
    ("Quartz Morning", "https://example.test/bulletins/quartz", "2026-06-01T09:30:00Z", "North Wire", "Line one.\nLine two."),
    ("Cedar Signal", "/bulletins/cedar", "2026-06-01T11:15:00Z", "Field Desk", None),
    ("Tidal Lantern", "/bulletins/tidal", "2026-06-01T13:45:00Z", "Coast Wire", "Current holds steady."),
    ("Mica Forecast", "https://example.test/bulletins/mica", "2026-06-01T16:10:00Z", "Hill Wire", "Visibility improves."),
    ("Amber Meridian", "/bulletins/amber", "2026-06-01T18:20:00Z", "South Wire", "Routes reopen."),
    ("Juniper Night", "/bulletins/juniper", "2026-06-01T21:05:00Z", "Field Desk", "Cool air arrives."),
]
PRISM_VARIANT = [
    ("Orchid Current", "/bulletins/orchid", "2026-06-04T03:00:00Z", "Delta Wire", "Channels converge."),
    ("Pewter Dawn", "/bulletins/pewter", "2026-06-04T05:00:00Z", "Delta Wire", None),
]

ORBIT_BASE_1 = [
    ("AX1", "Moss Dial", 14),
    ("AX2", "Reed Gauge", "27"),
    ("AX3", "Stone Vane", 5),
    ("AX4", "Cloud Pin", "103"),
    ("AX5", "Tern Ring", 42),
    ("AX6", "Fern Relay", "8"),
]
ORBIT_BASE_2 = [
    ("AX4", "Cloud Pin", "103"),
    ("AX7", "Opal Valve", 64),
    ("AX8", "Silt Beacon", "19"),
    ("AX9", "Linen Scope", 3),
    ("AX10", "Cinder Meter", "51"),
]
ORBIT_VARIANT_1 = [("NV1", "Aster Dial", "6"), ("NV2", "Birch Gauge", 71)]
ORBIT_VARIANT_2 = [("NV2", "Birch Gauge", "71"), ("NV3", "Coral Vane", 9)]


def _cards(records: list[tuple[str, str, str, int, str]]) -> bytes:
    blocks = []
    for sigil, label, group, rank, link in records:
        blocks.append(
            f'<article class="velin-shell other" data-glyph="{sigil}">'
            f'<h3 class="morrow-ink"> {escape(label)} </h3>'
            f'<span class="halo-band">{group}</span><i class="index-flare"> {rank} </i>'
            f'<a class="path-knot" href="{link}">open</a></article>'
        )
    return ("<!doctype html><aside class='velin-shell'>ignore</aside>" + "".join(blocks)).encode()


def _table(records: list[tuple[str, str, str, str]]) -> bytes:
    rows = ["<table id='noise-grid'><tr><td>discard</td></tr></table><table id='lithic-grid'><tbody>"]
    for index, (sector, station, reading, observed) in enumerate(records):
        cells = {
            "sector": f"<td data-col='sector'> {sector} </td>",
            "station": f"<td data-col='station'><span>{escape(station.split()[0])}</span> {' '.join(station.split()[1:])}</td>",
            "reading": f"<td data-col='reading'> {reading} </td>",
            "observed_at": f"<td data-col='observed_at'><time>{observed}</time></td>",
        }
        order = ("observed_at", "station", "sector", "reading") if index % 2 else ("sector", "reading", "station", "observed_at")
        rows.append("<tr data-entry='yes'><td class='decor'>x</td>" + "".join(cells[key] for key in order) + "</tr>")
    rows.append("<tr><td data-col='sector'>irrelevant</td></tr></tbody></table>")
    return "".join(rows).encode()


def _ledger(records: list[tuple[str, str, str]], next_href: str | None) -> bytes:
    chunks = ["<main><div class='folio-decoy'>skip</div>"]
    for mark, caption, amount in records:
        chunks.append(
            f"<section class='folio-slip' data-mark='{mark}'><b class='folio-caption'>{escape(caption)}</b>"
            f"<span class='folio-amount'>{amount}</span>"
            f"<a class='folio-source' href='/marks/{mark.lower()}'>source</a></section>"
        )
    if next_href:
        chunks.append(f"<a rel='next' class='folio-turn' href='{next_href}'>more</a>")
    chunks.append("</main>")
    return "".join(chunks).encode()


def _rss(records: list[tuple[str, str, str, str, str | None]]) -> bytes:
    items = []
    for headline, link, published, source, summary in records:
        summary_xml = "" if summary is None else f"<description>{summary}</description>"
        items.append(
            f"<item><title>{headline}</title><link>{link}</link><pubDate>{published}</pubDate>"
            f"<dc:creator>{escape(source)}</dc:creator>{summary_xml}</item>"
        )
    xml = (
        "<?xml version='1.0' encoding='UTF-8'?><rss version='2.0' xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        "<channel><title>Prism Dispatch</title>" + "".join(items) + "</channel></rss>"
    )
    return xml.encode()


def _orbit(records: list[tuple[str, str, object]], next_cursor: str | None) -> bytes:
    return json.dumps(
        {
            "samples": [
                {"identity": identity, "title": title, "measure": measure, "origin": f"/sensors/{identity.lower()}"}
                for identity, title, measure in records
            ],
            "next_cursor": next_cursor,
        }
    ).encode()


class FixtureState:
    def __init__(self, case: str) -> None:
        self.case = case
        self.requests: list[str] = []
        self.lock = threading.Lock()

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()

    def record(self, target: str) -> None:
        with self.lock:
            self.requests.append(target)

    def snapshot(self) -> list[str]:
        with self.lock:
            return list(self.requests)


def response_for(case: str, target: str) -> tuple[int, str, bytes]:
    parsed = urlsplit(target)
    query = parse_qs(parsed.query)
    variant = query.get("variant", ["1"])[0] == "2"
    if case == "opal_canopy" and parsed.path == "/aurora/cards":
        return 200, "text/html; charset=utf-8", _cards(AURORA_VARIANT if variant else AURORA_BASE)
    if case == "flint_harbor" and parsed.path == "/kiln/observations":
        return 200, "text/html; charset=utf-8", _table(KILN_VARIANT if variant else KILN_BASE)
    if case == "marble_tide" and parsed.path == "/ledger/start":
        records = LEDGER_VARIANT_1 if variant else LEDGER_BASE_1
        href = "/ledger/next?variant=2" if variant else "/ledger/next"
        return 200, "text/html; charset=utf-8", _ledger(records, href)
    if case == "marble_tide" and parsed.path == "/ledger/next":
        return 200, "text/html; charset=utf-8", _ledger(LEDGER_VARIANT_2 if variant else LEDGER_BASE_2, None)
    if case == "violet_prism" and parsed.path == "/prism/feed":
        return 200, "application/rss+xml; charset=utf-8", _rss(PRISM_VARIANT if variant else PRISM_BASE)
    if case == "copper_orbit" and parsed.path == "/orbit/measurements":
        cursor = query.get("cursor", [""])[0]
        if not cursor:
            return 200, "application/json", _orbit(ORBIT_VARIANT_1 if variant else ORBIT_BASE_1, "phase-violet-2" if variant else "phase-amber-2")
        expected = "phase-violet-2" if variant else "phase-amber-2"
        if cursor == expected:
            return 200, "application/json", _orbit(ORBIT_VARIANT_2 if variant else ORBIT_BASE_2, None)
    return 404, "text/plain; charset=utf-8", b"not found"


def make_handler(state: FixtureState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/__reset":
                state.reset()
                self._send(200, "application/json", b'{"reset":true}')
                return
            if self.path == "/__metrics":
                payload = json.dumps({"count": len(state.snapshot()), "requests": state.snapshot()}).encode()
                self._send(200, "application/json", payload)
                return
            state.record(self.path)
            self._send(*response_for(state.case, self.path))

        def _send(self, status: int, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


class FixtureServer(AbstractContextManager["FixtureServer"]):
    def __init__(self, case: str, host: str = "127.0.0.1", port: int = 0) -> None:
        self.state = FixtureState(case)
        self.server = ThreadingHTTPServer((host, port), make_handler(self.state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "FixtureServer":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(FixtureState(args.case)))
    server.serve_forever()


if __name__ == "__main__":
    main()

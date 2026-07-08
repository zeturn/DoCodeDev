from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from crawler import fetch_and_parse


EXPECTED_PRODUCTS = [
    {
        "id": "item-101",
        "name": "Trail Bottle",
        "url": "/shop/trail-bottle",
        "price": 18.75,
        "category": "Outdoors",
        "available": True,
    },
    {
        "id": "item-202",
        "name": "Canvas Tote",
        "url": "/shop/canvas-tote",
        "price": 14.5,
        "category": "Accessories",
        "available": False,
    },
]


class ProductSource:
    def __enter__(self):
        records = json.dumps({"products": EXPECTED_PRODUCTS}).encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/products.json":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(records)))
                self.end_headers()
                self.wfile.write(records)

            def log_message(self, format, *args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/products.json"
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class CrawlerTests(unittest.TestCase):
    def test_fetch_and_parse_reads_source_url(self):
        with ProductSource() as source:
            self.assertEqual(fetch_and_parse(source.url), EXPECTED_PRODUCTS)

    def test_cli_writes_json_output(self):
        with ProductSource() as source, tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "out.json"
            result = subprocess.run(
                [sys.executable, "crawler.py", source.url, "--output", str(output_path)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), EXPECTED_PRODUCTS)


if __name__ == "__main__":
    unittest.main()

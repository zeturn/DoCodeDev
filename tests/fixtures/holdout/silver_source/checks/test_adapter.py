import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from source_host import serve


class AdapterTests(unittest.TestCase):
    def test_local_http_records_are_written_with_requested_schema(self):
        server = serve()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "mosaic-result.json"
                completed = subprocess.run(
                    [sys.executable, "harvest_adapter.py", f"http://127.0.0.1:{server.server_port}/mosaic", str(target)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertEqual(
                    json.loads(target.read_text(encoding="utf-8")),
                    [
                        {"ember_code": "E-17", "caption": "Aster Vale", "drift_index": 9},
                        {"ember_code": "E-42", "caption": "Brass Willow", "drift_index": 14},
                    ],
                )
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

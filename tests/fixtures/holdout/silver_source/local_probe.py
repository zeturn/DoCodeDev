import subprocess
import sys
import threading

from source_host import serve


server = serve()
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    completed = subprocess.run(
        [sys.executable, "harvest_adapter.py", f"http://127.0.0.1:{server.server_port}/mosaic", "mosaic-result.json"],
        check=False,
    )
    raise SystemExit(completed.returncode)
finally:
    server.shutdown()
    server.server_close()

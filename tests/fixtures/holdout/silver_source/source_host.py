from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PAGE = b"""<!doctype html><html><body>
<section class="vanta-node" data-ember="E-17"><b class="lumen">Aster Vale</b><i class="drift">9</i></section>
<section class="vanta-node" data-ember="E-42"><b class="lumen">Brass Willow</b><i class="drift">14</i></section>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/mosaic":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)

    def log_message(self, *_args):
        pass


def serve(port=0):
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)

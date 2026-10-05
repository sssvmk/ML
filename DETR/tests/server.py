"""Tiny local HTTP server with Range support, request counters and a 'drop the connection once' fault injector."""
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeHost:
    def __init__(self, files: dict[str, bytes], support_range: bool = True, flaky_once: tuple[str, ...] = ()):
        self.files, self.support_range = files, support_range
        self.flaky_pending = set(flaky_once)
        self.stats = {"GET": 0, "HEAD": 0, "RANGE": 0, "paths": []}
        host = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _serve(self, head):
                data = host.files.get(self.path)
                if data is None:
                    self.send_error(404)
                    return
                host.stats["HEAD" if head else "GET"] += 1
                if not head:
                    host.stats["paths"].append(self.path)
                rng, start, status = self.headers.get("Range"), 0, 200
                if rng and host.support_range:
                    host.stats["RANGE"] += 1
                    start = int(re.match(r"bytes=(\d+)-", rng).group(1))
                    if start >= len(data):
                        self.send_response(416)
                        self.end_headers()
                        return
                    status = 206
                body = data[start:]
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes" if host.support_range else "none")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.end_headers()
                if head:
                    return
                if self.path in host.flaky_pending and start == 0:
                    host.flaky_pending.discard(self.path)
                    self.wfile.write(body[: int(len(body) * 0.4)])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

            def do_GET(self):
                self._serve(False)

            def do_HEAD(self):
                self._serve(True)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()

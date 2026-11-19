"""Shared local HTTP server for the test suite."""

import ssl
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BODY = b"0123456789abcdef" * 256  # 4096 bytes

CERTS_DIR = Path(__file__).resolve().parent / "certs"
TLS_CERT = CERTS_DIR / "localhost.crt"
TLS_KEY = CERTS_DIR / "localhost.key"

# Timing knobs, generous enough for loaded CI machines.
STALL_SECONDS = 0.4
DRIP_INTERVAL_SECONDS = 0.05
DRIP_BYTES = 20
SLOW_REDIRECT_SECONDS = 0.35
SLOW_REDIRECT_HOPS = 5
# One more than the client's default redirect limit of 10.
CHAIN_REDIRECTS = 11


class StatefulServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with per-instance counters for assertions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state: dict[str, int] = {}
        self.connections_opened = 0
        self._lock = threading.Lock()

    def finish_request(self, request, client_address) -> None:
        with self._lock:
            self.connections_opened += 1
        super().finish_request(request, client_address)


class QuietHandler(BaseHTTPRequestHandler):
    """Base handler: HTTP/1.1, keep-alive capable, no request logging."""

    protocol_version = "HTTP/1.1"

    server: StatefulServer

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class Handler(QuietHandler):
    """Endpoints covering success and failure response shapes."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urllib.parse.urlsplit(self.path).path
        try:
            self._route(path)
        except OSError:
            # The client went away mid-response; nothing to report.
            self.close_connection = True

    def _route(self, path: str) -> None:
        if path == "/ok":
            self._send(200, BODY)
        elif path == "/echo-path":
            encoded = f"path={self.path}".encode()
            self._send(200, encoded)
        elif path == "/redirect":
            self._redirect("/ok")
        elif path.startswith("/chain/"):
            self._chain(path)
        elif path.startswith("/redirect-slow/"):
            self._slow_chain(path)
        elif path == "/redirect-ftp":
            self._redirect("ftp://example.com/file")
        elif path == "/not-found":
            self._send(404)
        elif path == "/chunked":
            self._chunked()
        elif path == "/truncated":
            self._truncated()
        elif path == "/drip":
            self._drip()
        elif path == "/stall-headers":
            time.sleep(STALL_SECONDS)
            self._send(200, BODY)
        elif path == "/stall-body":
            self._send_head(200, len(BODY))
            time.sleep(STALL_SECONDS)
            self.wfile.write(BODY)
        elif path == "/close-framed":
            self._close_framed()
        elif path == "/stale-keepalive":
            # Keeps the connection alive, then silently closes it: the client
            # must detect the stale pooled socket and retry once.
            self._send(200, BODY)
            self.close_connection = True
        elif path == "/first-ok-then-hang":
            self._first_ok_then_hang()
        else:
            self._send(500)

    # -- response shapes ----------------------------------------------------

    def _send(self, status: int, body: bytes = b"") -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_head(self, status: int, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(length))
        self.end_headers()

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _chain(self, path: str) -> None:
        # /chain/0 -> /chain/1 -> ... -> /ok: CHAIN_REDIRECTS hops in total.
        current = int(path.rsplit("/", 1)[1])
        if current >= CHAIN_REDIRECTS:
            self._send(200, BODY)
        else:
            self._redirect(f"/chain/{current + 1}")

    def _slow_chain(self, path: str) -> None:
        current = int(path.rsplit("/", 1)[1])
        time.sleep(SLOW_REDIRECT_SECONDS)
        if current >= SLOW_REDIRECT_HOPS:
            self._redirect("/ok")
        else:
            self._redirect(f"/redirect-slow/{current + 1}")

    def _chunked(self) -> None:
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for part in (BODY[:1000], BODY[1000:2500], BODY[2500:]):
            self.wfile.write(f"{len(part):X}\r\n".encode("ascii"))
            self.wfile.write(part + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")

    def _truncated(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "1000")
        self.end_headers()
        self.wfile.write(b"x" * 10)
        self.wfile.flush()
        self.close_connection = True

    def _drip(self) -> None:
        """Send the body one byte at a time to exercise deadline checks."""
        self._send_head(200, DRIP_BYTES)
        for _ in range(DRIP_BYTES):
            self.wfile.write(b"x")
            self.wfile.flush()
            time.sleep(DRIP_INTERVAL_SECONDS)

    def _close_framed(self) -> None:
        """No Content-Length and no chunking: the body ends at EOF."""
        self.send_response(200)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(BODY)
        self.close_connection = True

    def _first_ok_then_hang(self) -> None:
        with self.server._lock:
            hits = self.server.state.get("hang_hits", 0) + 1
            self.server.state["hang_hits"] = hits
        if hits == 1:
            self._send(200, BODY)
        else:
            time.sleep(20)
            self._send(200, BODY)


class TestServer:
    """Context-managed local server bound to a free port."""

    def __init__(
        self,
        handler_class: type[BaseHTTPRequestHandler] = Handler,
        *,
        tls: bool = False,
    ) -> None:
        self.httpd = StatefulServer(("127.0.0.1", 0), handler_class)
        if tls:
            if not TLS_CERT.exists() or not TLS_KEY.exists():
                raise FileNotFoundError("test TLS certificate fixture is missing")
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(TLS_CERT, TLS_KEY)
            self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self.httpd.server_port

    @property
    def connections_opened(self) -> int:
        return self.httpd.connections_opened

    @property
    def state(self) -> dict[str, int]:
        return self.httpd.state

    def url_for(self, path: str, *, tls: bool = False) -> str:
        scheme = "https" if tls else "http"
        return f"{scheme}://127.0.0.1:{self.port}{path}"

    def __enter__(self) -> "TestServer":
        self.thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

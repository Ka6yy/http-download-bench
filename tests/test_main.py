"""Integration and unit tests for main.py using only the standard library."""

import argparse
import io
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import main as benchmark

BODY = b"0123456789abcdef" * 256  # 4096 bytes


class _TestHandler(BaseHTTPRequestHandler):
    """Small local server covering success and failure response shapes."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, status: int, body: bytes = b"") -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/ok":
            self._send(200, BODY)
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/ok")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/not-found":
            self._send(404)
        elif self.path == "/chunked":
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in (BODY[:1000], BODY[1000:2500], BODY[2500:]):
                self.wfile.write(f"{len(part):X}\r\n".encode("ascii"))
                self.wfile.write(part + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        elif self.path == "/truncated":
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b"x" * 10)
            self.wfile.flush()
            self.close_connection = True
        elif self.path == "/slow":
            self.send_response(200)
            self.send_header("Content-Length", "10")
            self.end_headers()
            time.sleep(0.4)
            try:
                self.wfile.write(b"0123456789")
                self.wfile.flush()
            except OSError:
                pass
            self.close_connection = True
        else:
            self._send(500)


class BenchmarkTests(unittest.TestCase):
    """Unit and HTTP integration tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _TestHandler)
        cls.thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True,
        )
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path: str, timeout: float = 2.0) -> benchmark.Attempt:
        return benchmark.perform_request(self.base_url + path, timeout, 1)

    def test_successful_request_receives_complete_body(self) -> None:
        attempt = self.request("/ok")

        self.assertTrue(attempt.ok)
        self.assertEqual(attempt.status, 200)
        self.assertEqual(attempt.bytes_received, len(BODY))
        self.assertGreater(attempt.elapsed_seconds, 0)

    def test_redirect_is_followed(self) -> None:
        attempt = self.request("/redirect")

        self.assertTrue(attempt.ok)
        self.assertEqual(attempt.status, 200)
        self.assertEqual(attempt.bytes_received, len(BODY))

    def test_non_2xx_status_is_reported(self) -> None:
        attempt = self.request("/not-found")

        self.assertFalse(attempt.ok)
        self.assertEqual(attempt.status, 404)
        self.assertIn("HTTP 404", attempt.error or "")

    def test_truncated_fixed_length_body_is_detected(self) -> None:
        attempt = self.request("/truncated")

        self.assertFalse(attempt.ok)
        self.assertEqual(attempt.bytes_received, 10)
        self.assertIn("incomplete response body", attempt.error or "")
        self.assertIn("of 1,000 bytes", attempt.error or "")

    def test_chunked_response_is_fully_read(self) -> None:
        attempt = self.request("/chunked")

        self.assertTrue(attempt.ok)
        self.assertEqual(attempt.bytes_received, len(BODY))

    def test_slow_response_hits_timeout(self) -> None:
        started = time.monotonic()
        attempt = self.request("/slow", timeout=0.15)
        elapsed = time.monotonic() - started

        self.assertFalse(attempt.ok)
        self.assertIn("timeout", (attempt.error or "").lower())
        self.assertLess(elapsed, 2.0)

    def test_run_benchmark_returns_zero_when_all_requests_succeed(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = benchmark.run_benchmark(self.base_url + "/ok", 2, 2.0)

        self.assertEqual(code, 0)
        self.assertIn("2 / 2 (100.0%)", output.getvalue())

    def test_run_benchmark_returns_one_on_failure(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = benchmark.run_benchmark(self.base_url + "/not-found", 1, 2.0)

        self.assertEqual(code, 1)
        self.assertIn("Failure reasons:", output.getvalue())
        self.assertIn("HTTP 404", output.getvalue())

    def test_calculate_stats_uses_successful_attempts(self) -> None:
        attempts = [
            benchmark.Attempt(1, 200, 1.0, 1_000_000),
            benchmark.Attempt(2, 200, 3.0, 3_000_000),
        ]

        stats = benchmark.calculate_stats(attempts)

        self.assertEqual(stats.count, 2)
        self.assertEqual(stats.total_bytes, 4_000_000)
        self.assertAlmostEqual(stats.total_seconds, 4.0)
        self.assertAlmostEqual(stats.average_seconds, 2.0)
        self.assertAlmostEqual(stats.minimum_seconds, 1.0)
        self.assertAlmostEqual(stats.maximum_seconds, 3.0)
        self.assertAlmostEqual(stats.speed_bytes_s, 1_000_000.0)
        self.assertAlmostEqual(stats.speed_mb_s, 1.0)
        self.assertAlmostEqual(stats.speed_mbit_s, 8.0)

    def test_summary_groups_failure_reasons(self) -> None:
        attempts = [
            benchmark.Attempt(1, 200, 1.0, 1_000_000),
            benchmark.Attempt(2, 404, 0.1, 0, "HTTP 404 Not Found"),
            benchmark.Attempt(3, 404, 0.1, 0, "HTTP 404 Not Found"),
        ]
        output = io.StringIO()

        with redirect_stdout(output):
            benchmark.print_summary(attempts, 3)

        text = output.getvalue()
        self.assertIn("1 / 3 (33.3%)", text)
        self.assertIn("Failure reasons:", text)
        self.assertIn("2x  HTTP 404 Not Found", text)

    def test_cli_defaults_and_overrides(self) -> None:
        parser = benchmark.build_parser()

        defaults = parser.parse_args(["https://example.com/file.jpg"])
        overrides = parser.parse_args(
            ["https://example.com/file.jpg", "-n", "7", "--timeout", "12.5"]
        )

        self.assertEqual(defaults.requests, 10)
        self.assertEqual(defaults.timeout, 30.0)
        self.assertEqual(overrides.requests, 7)
        self.assertEqual(overrides.timeout, 12.5)

    def test_cli_rejects_invalid_requests_count(self) -> None:
        parser = benchmark.build_parser()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                parser.parse_args(["https://example.com/file.jpg", "-n", "0"])

        self.assertEqual(context.exception.code, 2)

    def test_url_validation_rejects_non_http_scheme(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            benchmark.http_url("ftp://example.com/file.jpg")


if __name__ == "__main__":
    unittest.main()

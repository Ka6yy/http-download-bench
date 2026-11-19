"""Integration and unit tests for httpbench.network."""

import email.message
import http.client
import socket
import ssl
import unittest
import uuid

from helpers import BODY, TestServer

from httpbench import network
from httpbench.network import (
    HttpClient,
    Phases,
    declared_content_length,
    describe_exception,
)


def _headers(*pairs: tuple[str, str]) -> email.message.Message:
    message = email.message.Message()
    for key, value in pairs:
        message[key] = value
    return message


class RequestTests(unittest.TestCase):
    """End-to-end request behaviour against a local server."""

    def request(self, url: str, timeout: float = 5.0, keep_alive: bool = False):
        client = HttpClient()
        try:
            return client.get(
                url, timeout_seconds=timeout, number=1, keep_alive=keep_alive
            )
        finally:
            client.close()

    def test_successful_request_receives_complete_body_with_phases(self) -> None:
        with TestServer() as server:
            attempt = self.request(server.url_for("/ok"))

        self.assertTrue(attempt.ok)
        self.assertEqual(attempt.status, 200)
        self.assertEqual(attempt.bytes_received, len(BODY))
        self.assertIsNone(attempt.error)
        self.assertEqual(attempt.redirects, 0)
        phases = attempt.phases
        self.assertIsNotNone(phases.dns_seconds)
        self.assertIsNotNone(phases.connect_seconds)
        self.assertIsNone(phases.tls_seconds)  # Plain http.
        self.assertGreater(phases.ttfb_seconds, 0.0)
        self.assertGreaterEqual(phases.body_seconds or 0.0, 0.0)
        phase_sum = (
            (phases.dns_seconds or 0.0)
            + (phases.connect_seconds or 0.0)
            + (phases.ttfb_seconds or 0.0)
            + (phases.body_seconds or 0.0)
        )
        self.assertLessEqual(phase_sum, attempt.elapsed_seconds + 0.005)

    def test_redirect_is_followed_and_counted(self) -> None:
        with TestServer() as server:
            attempt = self.request(server.url_for("/redirect"))

        self.assertTrue(attempt.ok)
        self.assertEqual(attempt.status, 200)
        self.assertEqual(attempt.redirects, 1)
        self.assertEqual(attempt.bytes_received, len(BODY))

    def test_redirect_chain_over_limit_fails(self) -> None:
        with TestServer() as server:
            attempt = self.request(server.url_for("/chain/0"))

        self.assertFalse(attempt.ok)
        self.assertEqual(attempt.error, "too many redirects (limit is 10)")

    def test_redirect_to_non_http_scheme_fails(self) -> None:
        with TestServer() as server:
            attempt = self.request(server.url_for("/redirect-ftp"))

        self.assertFalse(attempt.ok)
        self.assertIn("redirect to unsupported URL", attempt.error or "")

    def test_non_2xx_status_is_reported(self) -> None:
        with TestServer() as server:
            attempt = self.request(server.url_for("/not-found"))

        self.assertFalse(attempt.ok)
        self.assertEqual(attempt.status, 404)
        self.assertIn("HTTP 404", attempt.error or "")

    def test_truncated_fixed_length_body_is_detected(self) -> None:
        with TestServer() as server:
            attempt = self.request(server.url_for("/truncated"))

        self.assertFalse(attempt.ok)
        self.assertEqual(attempt.bytes_received, 10)
        self.assertIn("incomplete response body", attempt.error or "")
        self.assertIn("of 1,000 bytes", attempt.error or "")

    def test_chunked_response_is_fully_read(self) -> None:
        with TestServer() as server:
            attempt = self.request(server.url_for("/chunked"))

        self.assertTrue(attempt.ok)
        self.assertEqual(attempt.bytes_received, len(BODY))

    def test_body_without_content_length_ends_at_eof(self) -> None:
        with TestServer() as server:
            attempt = self.request(server.url_for("/close-framed"))

        self.assertTrue(attempt.ok)
        self.assertEqual(attempt.bytes_received, len(BODY))
        self.assertIsNone(attempt.error)

    def test_drip_feed_body_hits_deadline_mid_body(self) -> None:
        # The server sends one byte every 50 ms, so no single read blocks for
        # the whole timeout: only the deadline loop check can stop it.
        with TestServer() as server:
            attempt = self.request(server.url_for("/drip"), timeout=0.3)

        self.assertFalse(attempt.ok)
        self.assertEqual(attempt.error, "timeout after 0.3 s")
        self.assertTrue(0 < attempt.bytes_received < 10)
        self.assertGreaterEqual(attempt.elapsed_seconds, 0.2)
        self.assertLess(attempt.elapsed_seconds, 2.0)

    def test_stalled_body_hits_socket_timeout(self) -> None:
        # Headers arrive immediately, then the server stalls before the body:
        # the blocking read itself must time out with the remaining budget.
        with TestServer() as server:
            attempt = self.request(server.url_for("/stall-body"), timeout=0.15)

        self.assertFalse(attempt.ok)
        self.assertEqual(attempt.error, "timeout after 0.15 s")
        self.assertEqual(attempt.bytes_received, 0)
        self.assertGreaterEqual(attempt.elapsed_seconds, 0.12)
        self.assertLess(attempt.elapsed_seconds, 2.0)

    def test_stalled_headers_hit_timeout(self) -> None:
        with TestServer() as server:
            attempt = self.request(server.url_for("/stall-headers"), timeout=0.15)

        self.assertFalse(attempt.ok)
        self.assertEqual(attempt.error, "timeout after 0.15 s")
        self.assertGreaterEqual(attempt.elapsed_seconds, 0.12)
        self.assertLess(attempt.elapsed_seconds, 2.0)

    def test_deadline_covers_the_whole_redirect_chain(self) -> None:
        # Five hops, each stalling 0.35 s. With a 0.6 s total budget the
        # request must stop near 0.6 s, not near 5 * 0.35 s.
        with TestServer() as server:
            attempt = self.request(server.url_for("/redirect-slow/0"), timeout=0.6)

        self.assertFalse(attempt.ok)
        self.assertEqual(attempt.error, "timeout after 0.6 s")
        self.assertGreaterEqual(attempt.elapsed_seconds, 0.45)
        self.assertLess(attempt.elapsed_seconds, 1.5)

    def test_connection_refused_is_reported(self) -> None:
        server = TestServer()
        server.thread.start()
        port = server.port
        server.httpd.shutdown()
        server.httpd.server_close()
        server.thread.join(timeout=5)

        attempt = self.request(f"http://127.0.0.1:{port}/ok")

        self.assertFalse(attempt.ok)
        self.assertIn("connection error", attempt.error or "")

    def test_dns_failure_is_reported(self) -> None:
        host = f"no-such-host-{uuid.uuid4().hex}.invalid"
        attempt = self.request(f"http://{host}/ok")

        self.assertFalse(attempt.ok)
        self.assertIn("DNS error", attempt.error or "")

    def test_self_signed_https_certificate_is_rejected(self) -> None:
        try:
            with TestServer(tls=True) as server:
                attempt = self.request(server.url_for("/ok", tls=True), timeout=5.0)
        except FileNotFoundError:
            self.skipTest("TLS certificate fixture is missing")

        self.assertFalse(attempt.ok)
        self.assertIn("TLS error", attempt.error or "")
        self.assertIn("CERTIFICATE_VERIFY_FAILED", attempt.error or "")

    def test_keep_alive_reuses_one_connection(self) -> None:
        with TestServer() as server:
            client = HttpClient()
            try:
                attempts = [
                    client.get(
                        server.url_for("/ok"),
                        timeout_seconds=5.0,
                        number=number,
                        keep_alive=True,
                    )
                    for number in (1, 2, 3)
                ]
            finally:
                client.close()

        self.assertTrue(all(attempt.ok for attempt in attempts))
        self.assertEqual(server.connections_opened, 1)
        self.assertFalse(attempts[0].reused)
        self.assertTrue(attempts[1].reused)
        self.assertTrue(attempts[2].reused)
        # Reused connections skip DNS, TCP and TLS entirely.
        self.assertIsNone(attempts[1].phases.dns_seconds)
        self.assertIsNone(attempts[1].phases.connect_seconds)
        self.assertIsNotNone(attempts[1].phases.ttfb_seconds)

    def test_without_keep_alive_each_request_connects(self) -> None:
        with TestServer() as server:
            client = HttpClient()
            try:
                attempts = [
                    client.get(
                        server.url_for("/ok"),
                        timeout_seconds=5.0,
                        number=number,
                    )
                    for number in (1, 2, 3)
                ]
            finally:
                client.close()

        self.assertTrue(all(attempt.ok for attempt in attempts))
        self.assertEqual(server.connections_opened, 3)
        self.assertTrue(all(not attempt.reused for attempt in attempts))
        self.assertIsNotNone(attempts[1].phases.dns_seconds)

    def test_keep_alive_survives_stale_connection(self) -> None:
        # The server closes the pooled connection without advertising it, so
        # the second request fails once and is retried on a fresh connection.
        with TestServer() as server:
            client = HttpClient()
            try:
                attempts = [
                    client.get(
                        server.url_for("/stale-keepalive"),
                        timeout_seconds=5.0,
                        number=number,
                        keep_alive=True,
                    )
                    for number in (1, 2)
                ]
            finally:
                client.close()

        self.assertTrue(all(attempt.ok for attempt in attempts))
        self.assertEqual(attempts[1].bytes_received, len(BODY))
        self.assertGreaterEqual(server.connections_opened, 2)


class ContentLengthTests(unittest.TestCase):
    """declared_content_length edge cases."""

    def test_missing_header_returns_none(self) -> None:
        self.assertIsNone(declared_content_length(_headers()))

    def test_single_value(self) -> None:
        self.assertEqual(
            declared_content_length(_headers(("Content-Length", "100"))), 100
        )

    def test_identical_duplicate_values(self) -> None:
        headers = _headers(("Content-Length", "100, 100"))
        self.assertEqual(declared_content_length(headers), 100)

    def test_conflicting_values_are_ignored(self) -> None:
        headers = _headers(("Content-Length", "100, 200"))
        self.assertIsNone(declared_content_length(headers))

    def test_negative_value_is_ignored(self) -> None:
        headers = _headers(("Content-Length", "-5"))
        self.assertIsNone(declared_content_length(headers))

    def test_garbage_value_is_ignored(self) -> None:
        headers = _headers(("Content-Length", "about ten"))
        self.assertIsNone(declared_content_length(headers))

    def test_chunked_responses_are_not_length_checked(self) -> None:
        headers = _headers(
            ("Transfer-Encoding", "chunked"),
            ("Content-Length", "100"),
        )
        self.assertIsNone(declared_content_length(headers))


class DescribeExceptionTests(unittest.TestCase):
    """Exception-to-message mapping."""

    def test_timeout(self) -> None:
        message = describe_exception(TimeoutError("timed out"), 2.0)
        self.assertEqual(message, "timeout after 2 s")

    def test_remote_disconnected(self) -> None:
        message = describe_exception(http.client.RemoteDisconnected(), 2.0)
        self.assertEqual(
            message, "connection closed before response headers were received"
        )

    def test_http_protocol_error(self) -> None:
        message = describe_exception(http.client.BadStatusLine("garbage"), 2.0)
        self.assertEqual(message, "HTTP protocol error: garbage")

    def test_dns_error(self) -> None:
        message = describe_exception(socket.gaierror("name or service not known"), 2.0)
        self.assertEqual(message, "DNS error: name or service not known")

    def test_tls_error(self) -> None:
        message = describe_exception(ssl.SSLEOFError("EOF occurred"), 2.0)
        self.assertTrue(message.startswith("TLS error:"))

    def test_connection_error(self) -> None:
        message = describe_exception(ConnectionRefusedError("refused"), 2.0)
        self.assertEqual(message, "connection error: refused")

    def test_generic_os_error(self) -> None:
        message = describe_exception(OSError("boom"), 2.0)
        self.assertEqual(message, "network error: boom")

    def test_unknown_exception_falls_back_to_type_name(self) -> None:
        message = describe_exception(RuntimeError("weird"), 2.0)
        self.assertEqual(message, "RuntimeError: weird")


class UtilityTests(unittest.TestCase):
    def test_format_error_collapses_and_truncates(self) -> None:
        long_error = "x" * 500
        formatted = network.format_error(long_error)
        self.assertEqual(len(formatted), network.MAX_ERROR_LENGTH)
        self.assertTrue(formatted.endswith("..."))

    def test_format_error_collapses_whitespace(self) -> None:
        self.assertEqual(network.format_error("a\n  b\tc"), "a b c")

    def test_phases_defaults_are_none(self) -> None:
        phases = Phases()
        self.assertIsNone(phases.dns_seconds)
        self.assertIsNone(phases.body_seconds)

    def test_is_success_status(self) -> None:
        for status in (200, 201, 204, 299):
            self.assertTrue(network.is_success_status(status))
        for status in (199, 300, 404, 500, None):
            self.assertFalse(network.is_success_status(status))


if __name__ == "__main__":
    unittest.main()

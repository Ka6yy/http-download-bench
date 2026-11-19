"""Deadline-aware HTTP client built directly on socket, ssl and http.client.

``urllib.request`` hides the connection inside handler objects, which makes it
impossible to time DNS, TCP and TLS separately, to reuse connections, or to
enforce one total timeout budget across redirects. This module keeps the
standard-library-only constraint but drives ``http.client`` itself.
"""

import email.message
import http.client
import socket
import ssl
import time
import urllib.parse
from dataclasses import dataclass, field

# Read size balances system-call overhead against constant memory use.
READ_CHUNK_BYTES = 64 * 1024

USER_AGENT = "http-download-bench/2.0"

DEFAULT_MAX_REDIRECTS = 10
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

# Redirect bodies larger than this are not drained; the connection is dropped
# instead because reading megabytes of redirect body would distort timing.
MAX_REDIRECT_DRAIN_BYTES = 64 * 1024

MAX_ERROR_LENGTH = 180

BASE_HEADERS: dict[str, str] = {
    "User-Agent": USER_AGENT,
    "Accept": "*/*",
    # Ask intermediaries not to compress the body so byte counts and speed
    # describe the wire, not an encoding we would have to account for.
    "Accept-Encoding": "identity",
}


class RequestDeadlineExceeded(Exception):
    """The total wall-clock budget of one request was exhausted.

    The message is identical for every phase (connect, headers, body) so that
    grouped failure reasons in the summary stay meaningful.
    """

    def __init__(self, timeout_seconds: float, received: int = 0) -> None:
        super().__init__(f"timeout after {timeout_seconds:g} s")
        self.received = received


class IncompleteBodyError(Exception):
    """The connection ended before the complete body was read."""

    def __init__(self, received: int, expected: int | None = None) -> None:
        self.received = received
        self.expected = expected
        if expected is None:
            message = (
                f"incomplete response body: connection closed after {received:,} bytes"
            )
        else:
            message = (
                f"incomplete response body: received {received:,} of {expected:,} bytes"
            )
        super().__init__(message)


class UnexpectedStatus(Exception):
    """The final response did not have a 2xx status."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(f"HTTP {status} {reason}")
        self.status = status


class TooManyRedirects(Exception):
    """The redirect chain exceeded the configured limit."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"too many redirects (limit is {limit})")


class InvalidUrlError(Exception):
    """A URL (initial or redirect target) is not an absolute http(s) URL."""


@dataclass
class Phases:
    """Per-request timing breakdown in seconds.

    Fields stay ``None`` for phases that did not happen, e.g. DNS/TCP/TLS on
    a reused keep-alive connection. ``ttfb_seconds`` covers writing the
    request up to having received the response headers (curl's definition of
    time_starttransfer). ``body_seconds`` is the transfer window between the
    first and the last body byte.
    """

    dns_seconds: float | None = None
    connect_seconds: float | None = None
    tls_seconds: float | None = None
    ttfb_seconds: float | None = None
    body_seconds: float | None = None


def is_success_status(status: int | None) -> bool:
    """Return True only for HTTP 2xx statuses."""
    return status is not None and 200 <= status < 300


@dataclass(frozen=True)
class Attempt:
    """Result of one HTTP GET attempt."""

    number: int
    status: int | None
    elapsed_seconds: float
    bytes_received: int
    phases: Phases = field(default_factory=Phases)
    redirects: int = 0
    reused: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the response completed with a 2xx status."""
        return self.error is None and is_success_status(self.status)


def format_error(value: object) -> str:
    """Collapse an error to one line and keep progress rows readable."""
    text = " ".join(str(value).split())
    if not text:
        text = "unknown error"
    if len(text) > MAX_ERROR_LENGTH:
        text = text[: MAX_ERROR_LENGTH - 3].rstrip() + "..."
    return text


def describe_exception(exc: Exception, timeout_seconds: float) -> str:
    """Turn transport exceptions into short, user-facing messages."""
    if isinstance(exc, TimeoutError):  # socket.timeout is its 3.10+ alias.
        return f"timeout after {timeout_seconds:g} s"
    if isinstance(exc, http.client.RemoteDisconnected):
        return "connection closed before response headers were received"
    if isinstance(exc, http.client.HTTPException):
        return f"HTTP protocol error: {exc}"
    if isinstance(exc, socket.gaierror):
        return f"DNS error: {exc}"
    if isinstance(exc, ssl.SSLError):
        return f"TLS error: {exc}"
    if isinstance(exc, ConnectionError):
        return f"connection error: {exc}"
    if isinstance(exc, OSError):
        return f"network error: {exc}"
    return f"{type(exc).__name__}: {exc}"


def declared_content_length(headers: email.message.Message) -> int | None:
    """Extract Content-Length when the body has a fixed length.

    Chunked responses are validated by http.client, so Content-Length is not
    used for them. Invalid or ambiguous headers are ignored rather than
    guessed.
    """
    transfer_encoding = headers.get("Transfer-Encoding", "")
    if "chunked" in transfer_encoding.lower():
        return None

    raw_length = headers.get("Content-Length")
    if raw_length is None:
        return None

    try:
        values = {int(part.strip()) for part in raw_length.split(",")}
    except ValueError:
        return None

    if len(values) != 1:
        return None
    value = values.pop()
    return value if value >= 0 else None


def _failure(
    number: int,
    started: float,
    status: int | None,
    received: int,
    error: object,
) -> Attempt:
    """Build a failed attempt, freezing the elapsed time immediately."""
    return Attempt(
        number=number,
        status=status,
        elapsed_seconds=time.perf_counter() - started,
        bytes_received=received,
        error=format_error(error),
    )


def _merge_phase(total: float | None, addition: float | None) -> float | None:
    """Accumulate one phase across connection attempts, keeping None."""
    if addition is None:
        return total
    if total is None:
        return addition
    return total + addition


def _check_deadline(deadline: float, timeout_seconds: float) -> None:
    if time.perf_counter() >= deadline:
        raise RequestDeadlineExceeded(timeout_seconds)


def _apply_socket_timeout(
    sock: socket.socket, deadline: float, timeout_seconds: float
) -> None:
    """Give the socket exactly the remaining budget as its timeout."""
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise RequestDeadlineExceeded(timeout_seconds)
    sock.settimeout(remaining)


def _request_target(parts: urllib.parse.SplitResult) -> str:
    """Build the origin-form request target from URL components."""
    raw = urllib.parse.urlunsplit(("", "", parts.path or "/", parts.query, ""))
    # Encode non-ASCII bytes without touching already-percent-encoded input.
    return urllib.parse.quote(raw, safe="!#$&'()*+,/:;=?@~-._%")


class HttpClient:
    """Minimal HTTP client for benchmarking.

    One instance is meant to serve the whole run so that the TLS context (and
    its certificate bundle) is created exactly once, outside any timing. A
    single pooled connection is reused across requests only when the caller
    passes ``keep_alive=True``.
    """

    def __init__(self, max_redirects: int = DEFAULT_MAX_REDIRECTS) -> None:
        self._ssl_context = ssl.create_default_context()
        self._connection: http.client.HTTPConnection | None = None
        self._connection_key: tuple[str, str, int] | None = None
        self._max_redirects = max_redirects

    def close(self) -> None:
        self._drop_connection()

    def get(
        self,
        url: str,
        *,
        timeout_seconds: float,
        number: int = 1,
        keep_alive: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> Attempt:
        """Perform one GET and receive the complete response body.

        The clock starts before DNS and stops when the last body byte has
        been received. Redirects are followed manually so that they share the
        same total deadline as the initial request.
        """
        started = time.perf_counter()
        deadline = started + timeout_seconds
        phases = Phases()
        status: int | None = None
        received = 0
        redirects = 0
        reused_any = False

        try:
            current_url = url
            while True:
                response, hop, hop_reused = self._single_request(
                    current_url,
                    deadline,
                    timeout_seconds,
                    keep_alive,
                    extra_headers,
                )
                phases.dns_seconds = _merge_phase(phases.dns_seconds, hop.dns_seconds)
                phases.connect_seconds = _merge_phase(
                    phases.connect_seconds, hop.connect_seconds
                )
                phases.tls_seconds = _merge_phase(phases.tls_seconds, hop.tls_seconds)
                phases.ttfb_seconds = _merge_phase(
                    phases.ttfb_seconds, hop.ttfb_seconds
                )
                reused_any = reused_any or hop_reused

                if response.status in REDIRECT_STATUSES:
                    location = response.headers.get("Location")
                    if location:
                        self._drain_redirect_body(response, deadline, timeout_seconds)
                        redirects += 1
                        if redirects > self._max_redirects:
                            raise TooManyRedirects(self._max_redirects)
                        current_url = self._redirect_target(current_url, location)
                        continue
                break

            status = response.status
            if not is_success_status(status):
                # Read nothing and drop the connection: its state is unknown.
                self._drop_connection()
                raise UnexpectedStatus(status, response.reason)

            received, body_window = self._read_body(response, deadline, timeout_seconds)
            phases.body_seconds = body_window
            self._release_connection(response, keep_alive)

            return Attempt(
                number=number,
                status=status,
                elapsed_seconds=time.perf_counter() - started,
                bytes_received=received,
                phases=phases,
                redirects=redirects,
                reused=reused_any,
            )

        except RequestDeadlineExceeded as exc:
            self._drop_connection()
            return _failure(number, started, status, exc.received, exc)
        except IncompleteBodyError as exc:
            self._drop_connection()
            return _failure(number, started, status, exc.received, exc)
        except UnexpectedStatus as exc:
            return _failure(number, started, exc.status, 0, exc)
        except (TooManyRedirects, InvalidUrlError) as exc:
            self._drop_connection()
            return _failure(number, started, status, 0, exc)
        except (OSError, http.client.HTTPException) as exc:
            # Only transport-level failures are converted to attempt errors;
            # bugs in this program must crash loudly instead of hiding in the
            # results table as "TypeError: ...".
            self._drop_connection()
            return _failure(
                number,
                started,
                status,
                received,
                describe_exception(exc, timeout_seconds),
            )

    # -- request plumbing ---------------------------------------------------

    def _single_request(
        self,
        url: str,
        deadline: float,
        timeout_seconds: float,
        keep_alive: bool,
        extra_headers: dict[str, str] | None,
    ) -> tuple[http.client.HTTPResponse, Phases, bool]:
        """Send one request (following no redirects) and read its headers."""
        parts = urllib.parse.urlsplit(url)
        scheme = parts.scheme.lower()
        host = parts.hostname
        if scheme not in ("http", "https") or host is None:
            raise InvalidUrlError(f"not an absolute http(s) URL: {url!r}")
        try:
            port = parts.port or (443 if scheme == "https" else 80)
        except ValueError as exc:
            raise InvalidUrlError(f"URL has an invalid port: {url!r}") from exc
        target = _request_target(parts)

        headers = dict(BASE_HEADERS)
        if not keep_alive:
            headers["Connection"] = "close"
        if extra_headers:
            headers.update(extra_headers)

        hop = Phases()
        for retry in (0, 1):
            connection, reused, dial = self._acquire(
                scheme, host, port, deadline, timeout_seconds, keep_alive
            )
            hop.dns_seconds = _merge_phase(hop.dns_seconds, dial.dns_seconds)
            hop.connect_seconds = _merge_phase(
                hop.connect_seconds, dial.connect_seconds
            )
            hop.tls_seconds = _merge_phase(hop.tls_seconds, dial.tls_seconds)

            sock = connection.sock
            if sock is None:  # _dial always sets a socket; kept for mypy.
                self._drop_connection()
                raise http.client.HTTPException("connection has no socket")
            try:
                _apply_socket_timeout(sock, deadline, timeout_seconds)
                request_started = time.perf_counter()
                connection.request("GET", target, headers=headers)
                response = connection.getresponse()
            except TimeoutError as exc:
                self._drop_connection()
                raise RequestDeadlineExceeded(timeout_seconds) from exc
            except (OSError, http.client.HTTPException):
                self._drop_connection()
                if reused and retry == 0:
                    # The pooled connection went stale between requests (the
                    # server closed it on its side). No response bytes were
                    # received for this request yet, so one retry on a fresh
                    # connection is safe and keeps keep-alive runs clean.
                    continue
                raise
            hop.ttfb_seconds = time.perf_counter() - request_started
            return response, hop, reused
        raise http.client.HTTPException("unreachable: retry loop exhausted")

    def _acquire(
        self,
        scheme: str,
        host: str,
        port: int,
        deadline: float,
        timeout_seconds: float,
        keep_alive: bool,
    ) -> tuple[http.client.HTTPConnection, bool, Phases]:
        """Return a live connection for the target plus its dial phases."""
        key = (scheme, host, port)
        dial = Phases()
        if keep_alive and self._connection is not None and self._connection_key == key:
            pooled = self._connection
            if pooled.sock is not None:
                _apply_socket_timeout(pooled.sock, deadline, timeout_seconds)
                return pooled, True, dial
        self._drop_connection()
        connection = self._dial(scheme, host, port, deadline, timeout_seconds, dial)
        self._connection = connection
        self._connection_key = key
        return connection, False, dial

    def _dial(
        self,
        scheme: str,
        host: str,
        port: int,
        deadline: float,
        timeout_seconds: float,
        dial: Phases,
    ) -> http.client.HTTPConnection:
        """Resolve, connect and (for https) handshake, recording the phases."""
        _check_deadline(deadline, timeout_seconds)

        started = time.perf_counter()
        addr_infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        dial.dns_seconds = time.perf_counter() - started
        # getaddrinfo() cannot be interrupted; re-check the budget after it.
        _check_deadline(deadline, timeout_seconds)

        last_error: OSError | None = None
        for family, socktype, proto, _canonical, address in addr_infos:
            _check_deadline(deadline, timeout_seconds)
            candidate = socket.socket(family, socktype, proto)
            try:
                _apply_socket_timeout(candidate, deadline, timeout_seconds)
                started = time.perf_counter()
                candidate.connect(address)
                dial.connect_seconds = time.perf_counter() - started
            except TimeoutError as exc:
                candidate.close()
                raise RequestDeadlineExceeded(timeout_seconds) from exc
            except OSError as exc:
                candidate.close()
                last_error = exc
                continue

            if scheme == "https":
                _apply_socket_timeout(candidate, deadline, timeout_seconds)
                started = time.perf_counter()
                try:
                    candidate = self._ssl_context.wrap_socket(
                        candidate, server_hostname=host
                    )
                except TimeoutError as exc:
                    candidate.close()
                    raise RequestDeadlineExceeded(timeout_seconds) from exc
                except (OSError, ValueError):
                    candidate.close()
                    raise
                dial.tls_seconds = time.perf_counter() - started

            connection = http.client.HTTPConnection(host, port)
            connection.sock = candidate
            return connection

        if last_error is not None:
            raise last_error
        raise OSError(f"no address found for {host}:{port}")

    def _read_body(
        self,
        response: http.client.HTTPResponse,
        deadline: float,
        timeout_seconds: float,
    ) -> tuple[int, float]:
        """Stream the complete body; return (byte count, transfer window)."""
        expected = declared_content_length(response.headers)

        # read1() returns available data immediately instead of waiting for a
        # full 64 KiB buffer. This keeps the deadline responsive to slowly
        # trickled responses. Fall back for unusual response objects.
        read_chunk = getattr(response, "read1", None) or response.read

        received = 0
        first_byte_at: float | None = None
        last_byte_at: float | None = None
        while True:
            sock = self._connection.sock if self._connection is not None else None
            if sock is not None:
                _apply_socket_timeout(sock, deadline, timeout_seconds)
            else:
                _check_deadline(deadline, timeout_seconds)
            try:
                chunk = read_chunk(READ_CHUNK_BYTES)
            except TimeoutError as exc:
                raise RequestDeadlineExceeded(timeout_seconds, received) from exc
            except http.client.IncompleteRead as exc:
                raise IncompleteBodyError(
                    received + len(exc.partial), expected
                ) from exc
            except (ConnectionError, http.client.HTTPException) as exc:
                raise IncompleteBodyError(received, expected) from exc

            if not chunk:
                break
            now = time.perf_counter()
            if first_byte_at is None:
                first_byte_at = now
            last_byte_at = now
            received += len(chunk)

            # read1() does not flag a fixed-length body that ends early, so
            # stop once Content-Length is satisfied and validate below.
            if expected is not None and received >= expected:
                break

        finished = last_byte_at if last_byte_at is not None else time.perf_counter()
        window = 0.0 if first_byte_at is None else finished - first_byte_at
        if expected is not None and received < expected:
            raise IncompleteBodyError(received, expected)
        return received, window

    def _drain_redirect_body(
        self,
        response: http.client.HTTPResponse,
        deadline: float,
        timeout_seconds: float,
    ) -> None:
        """Consume a small redirect body so the connection stays reusable."""
        expected = declared_content_length(response.headers)
        if expected is not None and expected > MAX_REDIRECT_DRAIN_BYTES:
            self._drop_connection()
            return
        self._read_body(response, deadline, timeout_seconds)

    @staticmethod
    def _redirect_target(current_url: str, location: str) -> str:
        target = urllib.parse.urljoin(current_url, location)
        parts = urllib.parse.urlsplit(target)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
            raise InvalidUrlError(f"redirect to unsupported URL: {location!r}")
        try:
            _ = parts.port  # Touching port validates its syntax.
        except ValueError as exc:
            raise InvalidUrlError(
                f"redirect to URL with invalid port: {location!r}"
            ) from exc
        return target

    def _release_connection(
        self, response: http.client.HTTPResponse, keep_alive: bool
    ) -> None:
        """Keep the pooled connection only if both sides may reuse it."""
        if keep_alive and not response.will_close:
            return
        self._drop_connection()

    def _drop_connection(self) -> None:
        connection = self._connection
        self._connection = None
        self._connection_key = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
